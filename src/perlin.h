#ifndef PERLIN_H
#define PERLIN_H

#include <glm/glm.hpp>

#include <thrust/random.h>

extern int* dev_perm_x;
extern int* dev_perm_y;
extern int* dev_perm_z;

__host__ __device__ inline float random_float(thrust::default_random_engine& rng) {
	// Returns a random real in [0,1).
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
	return u01(rng);
}

__host__ __device__ inline float random_float(float min, float max, thrust::default_random_engine& rng) {
	// Returns a random real in [min,max).
	return min + (max - min) * random_float(rng);
}

__host__ __device__ inline static glm::vec3 random(float min, float max, thrust::default_random_engine& rng) {
	return glm::vec3(random_float(min, max, rng), random_float(min, max, rng), random_float(min, max, rng));
}

__host__ __device__ inline int random_int(int min, int max, thrust::default_random_engine& rng) {
	// Returns a random integer in [min,max].
	return static_cast<int>(random_float(min, max + 1, rng));
}

class perlin {
    public:
        __host__ __device__ perlin(thrust::default_random_engine& rng, int* perm_x, int* perm_y, int* perm_z) {
            for (int i = 0; i < point_count; ++i) {
                ranvec[i] = glm::normalize(random(-1.0f, 1.0f, rng));
            }

			perlin_generate_perm(rng, perm_x);
			perlin_generate_perm(rng, perm_y);
			perlin_generate_perm(rng, perm_z);
        }

        __host__ __device__ float noise(const glm::vec3& p, int* perm_x, int* perm_y, int* perm_z) const {
            auto u = p.x - glm::floor(p.x);
            auto v = p.y - glm::floor(p.y);
            auto w = p.z - glm::floor(p.z);
            auto i = static_cast<int>(glm::floor(p.x));
            auto j = static_cast<int>(glm::floor(p.y));
            auto k = static_cast<int>(glm::floor(p.z));
            glm::vec3 c[2][2][2];

            for (int di = 0; di < 2; di++)
                for (int dj = 0; dj < 2; dj++)
                    for (int dk = 0; dk < 2; dk++)
                        c[di][dj][dk] = ranvec[
                            perm_x[(i+di) & 255] ^
                            perm_y[(j+dj) & 255] ^
                            perm_z[(k+dk) & 255]];

            return perlin_interp(c, u, v, w);
        }

        __host__ __device__ float turb(const glm::vec3& p, int depth, int* perm_x, int* perm_y, int* perm_z) const {
            auto accum = 0.0;
            auto temp_p = p;
            auto weight = 1.0;

            for (int i = 0; i < depth; i++) {
                accum += weight * noise(temp_p, perm_x, perm_y, perm_z);
                weight *= 0.5;
                temp_p *= 2;
            }

            return glm:: abs(accum);
        }

    private:
        static const int point_count = 256;
        glm::vec3 ranvec[point_count];

        __host__ __device__ void perlin_generate_perm(thrust::default_random_engine& rng, int* p) {
            for (int i = 0; i < point_count; i++)
                p[i] = i;

            permute(p, point_count, rng);
        }

        __host__ __device__ void permute(int* p, int n, thrust::default_random_engine& rng) {
            for (int i = n-1; i > 0; i--) {
                int target = random_int(0, i, rng);
                int tmp = p[i];
                p[i] = p[target];
                p[target] = tmp;
            }
        }

        __host__ __device__ static float perlin_interp(glm::vec3 c[2][2][2], float u, float v, float w) {
            auto uu = u * u * (3 - 2 * u);
            auto vv = v * v * (3 - 2 * v);
            auto ww = w * w * (3 - 2 * w);
            auto accum = 0.0;

            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    for (int k = 0; k < 2; k++) {
                        glm::vec3 weight_v(u - i, v - j, w - k);
                        accum += (i * uu + (1 - i) * (1 - uu)) *
                            (j * vv + (1 - j) * (1 - vv)) *
                            (k * ww + (1 - k) * (1 - ww)) * glm::dot(c[i][j][k], weight_v);
                    }

            return accum;
        }
};

#endif
