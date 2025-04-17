#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <float.h>

#define MAX_ITER 100

typedef struct {
    double x, y;
    int cluster;
} Point;

typedef struct {
    double x, y;
    int count;
} Centroid;

double euclidean_distance(Point a, Centroid b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

void generate_points(Point* points, int n) {
    for (int i = 0; i < n; i++) {
        points[i].x = rand() % 1000;
        points[i].y = rand() % 1000;
        points[i].cluster = -1;
    }
}

void initialize_centroids(Point* points, Centroid* centroids, int k) {
    for (int i = 0; i < k; i++) {
        centroids[i].x = points[i].x;
        centroids[i].y = points[i].y;
        centroids[i].count = 0;
    }
}

void kmeans_sequential(Point* points, Centroid* centroids, int n, int k) {
    for (int iter = 0; iter < MAX_ITER; iter++) {
        int changed = 0;
        for (int i = 0; i < n; i++) {
            double min_dist = 1e9;
            int min_index = -1;
            for (int j = 0; j < k; j++) {
                double dist = euclidean_distance(points[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_index = j;
                }
            }
            if (points[i].cluster != min_index) {
                points[i].cluster = min_index;
                changed = 1;
            }
        }

        for (int j = 0; j < k; j++) {
            centroids[j].x = 0;
            centroids[j].y = 0;
            centroids[j].count = 0;
        }

        for (int i = 0; i < n; i++) {
            int cl = points[i].cluster;
            centroids[cl].x += points[i].x;
            centroids[cl].y += points[i].y;
            centroids[cl].count++;
        }

        for (int j = 0; j < k; j++) {
            if (centroids[j].count > 0) {
                centroids[j].x /= centroids[j].count;
                centroids[j].y /= centroids[j].count;
            }
        }

        if (!changed) break;
    }
}

void kmeans_parallel(Point* points, Centroid* centroids, int n, int k, int num_threads) {
    omp_set_num_threads(num_threads);

    for (int iter = 0; iter < MAX_ITER; iter++) {
        int changed = 0;

        #pragma omp parallel for reduction(+:changed)
        for (int i = 0; i < n; i++) {
            double min_dist = 1e9;
            int min_index = -1;

            for (int j = 0; j < k; j++) {
                double dist = euclidean_distance(points[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_index = j;
                }
            }

            if (points[i].cluster != min_index) {
                points[i].cluster = min_index;
                changed++;
            }
        }

        Centroid* temp = calloc(k, sizeof(Centroid));

        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            int cl = points[i].cluster;
            #pragma omp atomic
            temp[cl].x += points[i].x;
            #pragma omp atomic
            temp[cl].y += points[i].y;
            #pragma omp atomic
            temp[cl].count++;
        }

        for (int j = 0; j < k; j++) {
            if (temp[j].count > 0) {
                centroids[j].x = temp[j].x / temp[j].count;
                centroids[j].y = temp[j].y / temp[j].count;
            }
        }

        free(temp);
        if (changed == 0) break;
    }
}

void run_elbow_method_parallel(Point* original_points, int n_points, int k_min, int k_max, int threads) {
    printf(">>> MÉTODO DO COTOVELO (paralelo com %d threads) <<<\n", threads);

    omp_set_num_threads(threads);
    int range = k_max - k_min + 1;
    double* inertias = malloc(sizeof(double) * range);

    double start = omp_get_wtime();

    #pragma omp parallel for
    for (int idx = 0; idx < range; idx++) {
        int k = k_min + idx;

        Point* points = malloc(sizeof(Point) * n_points);
        Centroid* centroids = malloc(sizeof(Centroid) * k);

        for (int i = 0; i < n_points; i++) points[i] = original_points[i];
        initialize_centroids(points, centroids, k);

        kmeans_parallel(points, centroids, n_points, k, threads);

        double inertia = 0.0;
        for (int i = 0; i < n_points; i++) {
            int cl = points[i].cluster;
            double dx = points[i].x - centroids[cl].x;
            double dy = points[i].y - centroids[cl].y;
            inertia += dx * dx + dy * dy;
        }

        inertias[idx] = inertia;

        free(points);
        free(centroids);
    }

    for (int idx = 0; idx < range; idx++) {
        printf("k = %3d | Inércia = %.2f\n", k_min + idx, inertias[idx]);
    }

    int best_k = k_min;
    double max_distance = -DBL_MAX;

    double x1 = k_min, y1 = inertias[0];
    double x2 = k_max, y2 = inertias[range - 1];

    for (int idx = 0; idx < range; idx++) {
        double x0 = k_min + idx;
        double y0 = inertias[idx];

        double numerator = fabs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1);
        double denominator = sqrt((y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1));
        double distance = numerator / denominator;

        if (distance > max_distance) {
            max_distance = distance;
            best_k = x0;
        }
    }

    double end = omp_get_wtime();
    printf("Tempo total: %.4f segundos\n", end - start);
    printf(">> Melhor k sugerido (cotovelo): %d <<\n\n", best_k);

    free(inertias);
}

void run_elbow_method_sequential(Point* original_points, int n_points, int k_min, int k_max) {
    printf(">>> MÉTODO DO COTOVELO SEQUENCIAL <<<\n");

    int range = k_max - k_min + 1;
    double* inertias = malloc(sizeof(double) * range);

    double start = omp_get_wtime();

    for (int idx = 0; idx < range; idx++) {
        int k = k_min + idx;

        Point* points = malloc(sizeof(Point) * n_points);
        Centroid* centroids = malloc(sizeof(Centroid) * k);

        for (int i = 0; i < n_points; i++) points[i] = original_points[i];
        initialize_centroids(points, centroids, k);

        kmeans_sequential(points, centroids, n_points, k);

        double inertia = 0.0;
        for (int i = 0; i < n_points; i++) {
            int cl = points[i].cluster;
            double dx = points[i].x - centroids[cl].x;
            double dy = points[i].y - centroids[cl].y;
            inertia += dx * dx + dy * dy;
        }

        inertias[idx] = inertia;

        free(points);
        free(centroids);
    }

    for (int idx = 0; idx < range; idx++) {
        printf("k = %3d | Inércia = %.2f\n", k_min + idx, inertias[idx]);
    }

    int best_k = k_min;
    double max_distance = -DBL_MAX;

    double x1 = k_min, y1 = inertias[0];
    double x2 = k_max, y2 = inertias[range - 1];

    for (int idx = 0; idx < range; idx++) {
        double x0 = k_min + idx;
        double y0 = inertias[idx];

        double numerator = fabs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1);
        double denominator = sqrt((y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1));
        double distance = numerator / denominator;

        if (distance > max_distance) {
            max_distance = distance;
            best_k = x0;
        }
    }

    double end = omp_get_wtime();
    printf("Tempo total: %.4f segundos\n", end - start);
    printf(">> Melhor k sugerido (cotovelo): %d <<\n\n", best_k);

    free(inertias);
}

void run_strong_scalability_with_elbow(int n_points, int k_min, int k_max) {
    printf(">>> ESCALABILIDADE FORTE + COTOVELO <<<\n");

    Point* base_points = malloc(sizeof(Point) * n_points);
    generate_points(base_points, n_points);

    int threads_list[] = {1, 2, 4, 8, 16};
    for (int i = 0; i < 5; i++) {
        int threads = threads_list[i];
        printf("[FORTE] Threads: %d\n", threads);
        if (threads == 1)
            run_elbow_method_sequential(base_points, n_points, k_min, k_max);
        else
            run_elbow_method_parallel(base_points, n_points, k_min, k_max, threads);
    }

    free(base_points);
}

void run_weak_scalability_with_elbow(int base_points_per_thread, int k_min, int k_max) {
    printf(">>> ESCALABILIDADE FRACA + COTOVELO <<<\n");

    int threads_list[] = {1, 2, 4, 8, 16};
    for (int i = 0; i < 5; i++) {
        int threads = threads_list[i];
        int total_points = base_points_per_thread * threads;

        Point* points = malloc(sizeof(Point) * total_points);
        generate_points(points, total_points);

        
        if (threads == 1){
            printf("[FRACA] Primeira Base Sequencial (total pontos = %d)\n", base_points_per_thread);
            run_elbow_method_sequential(points, total_points, k_min, k_max);

        }else{
            printf("[FRACA] Sequencial com (total pontos = %d)\n", total_points);
            run_elbow_method_sequential(points, total_points, k_min, k_max);
            printf("[FRACA] Paralelo com %d threads (total pontos = %d)\n", threads, total_points);
            run_elbow_method_parallel(points, total_points, k_min, k_max, threads);
        }
        free(points);
    }
}

int main() {
    int base_n = 100000;
    int base_points_per_thread = 100000;
    int k_min = 2, k_max = 10;

    run_strong_scalability_with_elbow(base_n, k_min, k_max);
    printf("\n");
    run_weak_scalability_with_elbow(base_points_per_thread, k_min, k_max);

    return 0;
}
