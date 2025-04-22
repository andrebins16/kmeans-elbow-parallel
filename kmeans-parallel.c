#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define MAX_ITER 100


typedef struct {
    double x, y;
    int cluster;
} Point;

typedef struct {
    double x, y;
    int count;    // numero de pontos atribuidos a esse centroide
} Centroid;

void save_results_to_file(Point* points, Centroid* centroids, int n_points, int k, int threads, double time, double inertia, const char* tipo) {
    // monta o nome do arquivo
    char filename[100];
    snprintf(filename, sizeof(filename), "resultados_%s_threads%d_pontos%d.txt", tipo, threads, n_points);

    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        perror("Erro ao abrir o arquivo para escrita");
        exit(1);
    }

    fprintf(file, "Tipo de escalabilidade: %s\n", tipo);
    fprintf(file, "Threads: %d\n", threads);
    fprintf(file, "Número de pontos: %d\n", n_points);
    fprintf(file, "Tempo de execução: %.4f segundos\n", time);
    fprintf(file, "Inércia: %.2f\n", inertia);

    fprintf(file, "\nCentroides finais:\n");
    for (int j = 0; j < k; j++) {
        fprintf(file, "Centroid %d: (%.2f, %.2f) com %d pontos\n", j, centroids[j].x, centroids[j].y, centroids[j].count);
    }

    fprintf(file, "\nPontos e seus clusters:\n");
    for (int i = 0; i < n_points; i++) {
        fprintf(file, "Ponto %d: (%.2f, %.2f) -> Cluster %d\n", i, points[i].x, points[i].y, points[i].cluster);
    }

    fclose(file);
}


double euclidean_distance(Point a, Centroid b) {
    return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}

void generate_points(Point* points, int n) {
    for (int i = 0; i < n; i++) {
        points[i].x = rand() % 1000000;
        points[i].y = rand() % 1000000;
        points[i].cluster = -1;  // pontos começam sem cluster
    }
}

// inicializa os primeiros pontos como centroides
void initialize_centroids(Point* points, Centroid* centroids, int k) {
    for (int i = 0; i < k; i++) {
        centroids[i].x = points[i].x;
        centroids[i].y = points[i].y;
        centroids[i].count = 0;  // nenhum ponto nesse centroide ainda
    }
}

// kmeans sequencial
void kmeans_sequential(Point* points, Centroid* centroids, int n, int k) {
    for (int iter = 0; iter < MAX_ITER; iter++) {
        int changed = 0;  // verifica se algum ponto mudou

        // atribui cada ponto ao centroide mais proximo
        for (int i = 0; i < n; i++) {
            double min_dist = 1e9;
            int min_index = -1;  // indice do cluster mais proximo
            for (int j = 0; j < k; j++) {
                double dist = euclidean_distance(points[i], centroids[j]);
                if (dist < min_dist) {  // centroide atual é o mais proximo
                    min_dist = dist;
                    min_index = j;
                }
            }
            if (points[i].cluster != min_index) {  // se mudou de cluster
                points[i].cluster = min_index;
                changed = 1;  // indica mudança para condição de parada
            }
        }

        // zera a soma dos centroides
        for (int j = 0; j < k; j++) {
            centroids[j].x = 0;
            centroids[j].y = 0;
            centroids[j].count = 0;
        }

        // soma as cordenadas dos pontos de cada centroide
        for (int i = 0; i < n; i++) {
            int cl = points[i].cluster;
            centroids[cl].x += points[i].x;
            centroids[cl].y += points[i].y;
            centroids[cl].count++;
        }

        // calcula novos centroides com a media
        for (int j = 0; j < k; j++) {
            if (centroids[j].count > 0) {
                centroids[j].x /= centroids[j].count;
                centroids[j].y /= centroids[j].count;
            }
        }

        // nenhum ponto mudou de cluster, entao termina
        if (!changed) break;
    }
}

// kmeans paralelo
void kmeans_parallel(Point* points, Centroid* centroids, int n, int k, int n_threads) {
    omp_set_num_threads(n_threads);  //define numero de threads

    for (int iter = 0; iter < MAX_ITER; iter++) {
        int changed = 0;  // verifica se algum ponto mudou

        // atribui cada ponto ao centroide mais proximo
        #pragma omp parallel for reduction(+:changed)  // paraleliza o loop, fazendo uma redução na variavel changed
        for (int i = 0; i < n; i++) {
            double min_dist = 1e9;
            int min_index = -1; // indice do cluster mais proximo
            for (int j = 0; j < k; j++) {
                double dist = euclidean_distance(points[i], centroids[j]);
                if (dist < min_dist) { // centroide atual é o mais proximo
                    min_dist = dist;
                    min_index = j;
                }
            }
            if (points[i].cluster != min_index) { //se mudou de cluster
                points[i].cluster = min_index;
                changed++;  // indica mudança para condição de parada
            }
        }

        // zera a soma dos centroides
        for (int j = 0; j < k; j++) {
            centroids[j].x = 0;
            centroids[j].y = 0;
            centroids[j].count = 0;
        }


        // abaixo, quebra o calculo dos novos centroides em threads. calcula a soma paralelamente em threads separadas, e depois combina os resultados

        // cria uma matriz para guardar temporariamente os resultados parciais de cada thread. local_temp[thread][centroid]
        Centroid (*local_temp)[k] = calloc(n_threads, sizeof(Centroid[k]));

        #pragma omp parallel  // cria regiao paralela para a soma das coordenadas dos pontos que cada thread vai processar
        {
            int tid = omp_get_thread_num();  // pega o id da thread
            Centroid* thread_centroids = local_temp[tid]; // array de centroides da thread tid

            #pragma omp for  // paraleliza o loop que soma as cordenadas dos pontos de cada centroide. cada thread vai ter suas somas parciais, relativas aos pontos que processou
            for (int i = 0; i < n; i++) {
                int cl = points[i].cluster;
                thread_centroids[cl].x += points[i].x;
                thread_centroids[cl].y += points[i].y;
                thread_centroids[cl].count++;
            }
        }

        // combina (soma) os resultados de todas as threads
        for (int t = 0; t < n_threads; t++) {
            for (int j = 0; j < k; j++) {
                centroids[j].x += local_temp[t][j].x;
                centroids[j].y += local_temp[t][j].y;
                centroids[j].count += local_temp[t][j].count;
            }
        }

        free(local_temp);

        // calcula novos centroides com a media
        for (int j = 0; j < k; j++) {
            if (centroids[j].count > 0) {
                centroids[j].x = centroids[j].x / centroids[j].count;
                centroids[j].y = centroids[j].y / centroids[j].count;
            }
        }

        // nenhum ponto mudou de cluster, entao termina
        if (changed == 0) break;
    }
}

void run_generic_kmeans(Point* original_points, int n_points, int k, int threads, const char* tipo) {
    Point* points = malloc(sizeof(Point) * n_points);
    Centroid* centroids = malloc(sizeof(Centroid) * k);

    for (int i = 0; i < n_points; i++) points[i] = original_points[i];

    initialize_centroids(points, centroids, k);

    double start = omp_get_wtime();

    if (threads == 1)
        kmeans_sequential(points, centroids, n_points, k);
    else
        kmeans_parallel(points, centroids, n_points, k, threads);

    double end = omp_get_wtime();
    double elapsed = end - start;

    // calcula inercia
    double inertia = 0.0;
    for (int i = 0; i < n_points; i++) {
        int cl = points[i].cluster;
        double dx = points[i].x - centroids[cl].x;
        double dy = points[i].y - centroids[cl].y;
        inertia += dx * dx + dy * dy;
    }

    printf("[Threads: %2d | Pontos: %d] Tempo: %.4f s | Inércia: %.2f\n", threads, n_points, elapsed, inertia);

    // salva no arquivo
    save_results_to_file(points, centroids, n_points, k, threads, elapsed, inertia, tipo);

    free(points);
    free(centroids);
}


// roda a escalabilidade forte (dados fixos com threads variando)
void run_strong_scalability(int n_points, int k) {
    printf(">>> ESCALABILIDADE FORTE <<<\n");

    Point* base_points = malloc(sizeof(Point) * n_points);
    generate_points(base_points, n_points);

    // roda o kmeans com diferentes numeros de threads
    run_generic_kmeans(base_points, n_points, k, 1,"forte");
    run_generic_kmeans(base_points, n_points, k, 2,"forte");
    run_generic_kmeans(base_points, n_points, k, 4,"forte");
    run_generic_kmeans(base_points, n_points, k, 8,"forte");
    run_generic_kmeans(base_points, n_points, k, 16,"forte");

    free(base_points);
}

// roda a escalabilidade fraca (número de pontos cresce com o número de threads)
void run_weak_scalability(int base_points_per_thread, int k) {
    printf(">>> ESCALABILIDADE FRACA <<<\n");

    int threads_list[] = {1, 2, 4, 8, 16};

    for (int i = 0; i < 5; i++) {
        int threads = threads_list[i];
        int total_points = base_points_per_thread * threads;

        Point* points = malloc(sizeof(Point) * total_points);
        generate_points(points, total_points);

        if (threads != 1) {
            run_generic_kmeans(points, total_points, k, 1,"fraca");  // sequencial para comparação
        }
        run_generic_kmeans(points, total_points, k, threads,"fraca");  // paralelo com n de threads atual
        printf("\n");
        free(points);
    }
}

int main() {
    int n = 3000000;
    int k = 4;

    run_strong_scalability(n, k);
    printf("\n");
    run_weak_scalability(n, k);

    return 0;
}