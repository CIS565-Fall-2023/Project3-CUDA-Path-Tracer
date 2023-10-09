#pragma once

#include <cuda_runtime.h>

#include <glm/glm.hpp>
#define Clamp(x, min, max) (x < min ? min : (x > max ? max : x))

namespace Math {
	__host__ __device__ inline float luminance(const glm::vec3 & color) {
		return glm::dot(color, glm::vec3(0.2126f, 0.7152f, 0.0722f));
	}

	/* Return a sampled barycentric coordinate for triangle */
	__host__ __device__ inline glm::vec2 uniformSampleTriangle(const glm::vec2& u) {
		float su0 = sqrtf(u.x);
		return glm::vec2(1 - su0, u.y * su0);
	}

#pragma region BSDF
    // BSDF Inline Functions
    __device__ inline float CosTheta(const glm::vec3& w) { return w.z; }
    __device__ inline float Cos2Theta(const glm::vec3& w) { return w.z * w.z; }
    __device__ inline float AbsCosTheta(const glm::vec3& w) { return std::abs(w.z); }
    __device__ inline float Sin2Theta(const glm::vec3& w) {
        return __max((float)0, (float)1 - Cos2Theta(w));
    }

    __device__ inline float SinTheta(const glm::vec3& w) { return std::sqrt(Sin2Theta(w)); }

    __device__ inline float TanTheta(const glm::vec3& w) { return SinTheta(w) / CosTheta(w); }

    __device__ inline float Tan2Theta(const glm::vec3& w) {
        return Sin2Theta(w) / Cos2Theta(w);
    }

    __device__ inline float CosPhi(const glm::vec3& w) {
        float sinTheta = SinTheta(w);
        return (sinTheta == 0) ? 1 : Clamp(w.x / sinTheta, -1, 1);
    }

    __device__ inline float SinPhi(const glm::vec3& w) {
        float sinTheta = SinTheta(w);
        return (sinTheta == 0) ? 0 : Clamp(w.y / sinTheta, -1, 1);
    }

    __device__ inline float Cos2Phi(const glm::vec3& w) { return CosPhi(w) * CosPhi(w); }

    __device__ inline float Sin2Phi(const glm::vec3& w) { return SinPhi(w) * SinPhi(w); }

    __device__ inline float CosDPhi(const glm::vec3& wa, const glm::vec3& wb) {
        float waxy = wa.x * wa.x + wa.y * wa.y;
        float wbxy = wb.x * wb.x + wb.y * wb.y;
        if (waxy == 0 || wbxy == 0)
            return 1;
        return Clamp((wa.x * wb.x + wa.y * wb.y) / std::sqrt(waxy * wbxy), -1, 1);
    }


    __device__ inline float Lambda(const glm::vec3& w, const float alpha) {
        float absTanTheta = std::abs(TanTheta(w));
        if (glm::isinf(absTanTheta)) return 0.;
        float alpha_lambda = glm::sqrt(Cos2Phi(w) +
            Sin2Phi(w)) * alpha;

        float a = 1 / (alpha_lambda * absTanTheta);
        if (a >= 1.6f)
            return 0;
        return (1 - 1.259f * a + 0.396f * a * a) /
            (3.535f * a + 2.181f * a * a);
    }

#pragma endregion

}