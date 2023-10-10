#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng,float exponent) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = pow(u01(rng),exponent); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__
float reflectance(float cosine, float ref_idx) {
    // Use Schlick's approximation for reflectance.
    float r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f-r0)*pow((1.0f - cosine),5.0f);
}

__host__ __device__
glm::vec3 refract(glm::vec3& uv, glm::vec3& n, float etai_over_etat) {
    float cos_theta = glm::min(dot(-uv, n), 1.0f);
    glm::vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    glm::vec3 r_out_parallel = -glm::sqrt(glm::abs(1.0f - glm::length2(r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}
__host__ __device__
void sampleLight(glm::vec3 &dist,Geom & light,
    float & lightsize,thrust::default_random_engine &rng,float &pdf,glm::vec3& orig){
    thrust::uniform_real_distribution<float> u01(0, 1);
    float r1=u01(rng);
    float r2=u01(rng);
    float r3=u01(rng);
    glm::vec3 point;
    if(light.type==TRIANGLE){
        float u=r2*r3;
        float v=r2*(1-r3);
        point=(1.0f-u-v)*light.tri.vertices[0]+u*light.tri.vertices[1]+v*light.tri.vertices[2];
    }else if(light.type==CUBE){
        point=glm::vec3(r1-0.5f,r2-0.5f,r3-0.5f);
    }else {
        float theta=r1* TWO_PI;
        float phi=r2* PI;
        point=glm::vec3(glm::cos(theta)*glm::sin(phi),glm::cos(phi),glm::sin(theta)*glm::sin(phi))*0.5f;
    }
    dist=multiplyMV(light.transform, glm::vec4(point, 1.f));
}
__host__ __device__
float computeG(glm::vec3 w,glm::vec3 normal,float expoenent){
    float cos=glm::dot(w,normal);
    float sin=glm::sqrt(1.0f-cos*cos);
    float a=glm::sqrt(0.5f* expoenent +1)/(sin/cos);
    if(a<1.6f){
        return (3.535*a+2.181*a*a)/(1.0+2.276*a+2.577*a*a);
    }else{
        return 1.0f;
    }
}

__host__ __device__
void diffuseScatter(PathSegment & pathSegment,
        ShadeableIntersection& intersection,
        glm::vec3 &materialColor,thrust::default_random_engine &rng, Geom & light,
    float & lightsize){
    float cos=glm::dot(-glm::normalize(pathSegment.ray.direction), intersection.surfaceNormal);
    thrust::uniform_real_distribution<float> u01(0, 1);
    float r2=u01(rng);
    pathSegment.color *= materialColor;
    pathSegment.remainingBounces--;
    pathSegment.ray.origin=getPointOnRay(pathSegment.ray, intersection.t);
    /*if(r2<0.5f){
        glm::vec3 dist;
        float pdf;
        sampleLight(dist,light,lightsize,rng,pdf,pathSegment.ray.origin);

        pathSegment.ray.direction=glm::normalize(dist-pathSegment.ray.origin);
        pathSegment.color*=glm::abs(glm::dot(pathSegment.ray.direction, intersection.surfaceNormal))/PI;
        pathSegment.remainingBounces=1;
        //float pdf=glm::length2(dist-pathSegment.ray.origin)/cos;
        //pathSegment.color/=pdf;
    }else{*/
        pathSegment.ray.direction=glm::normalize(calculateRandomDirectionInHemisphere(intersection.surfaceNormal,rng,1.0f));
    //}
    
}

__host__ __device__
void refractScatter(PathSegment & pathSegment,
        ShadeableIntersection& intersection,
     glm::vec3 &materialColor,const float ior,thrust::default_random_engine &rng){
    thrust::uniform_real_distribution<float> u01(0, 1);
    float refraction_ratio = intersection.outside ? (1.0f/ior):(ior);
    float cos_theta = glm::min(glm::dot(-glm::normalize(pathSegment.ray.direction), intersection.surfaceNormal), 1.0f);
    bool cannot_refract = ((glm::sqrt(1.0f - cos_theta * cos_theta)*refraction_ratio ) > 1.0f);
    
    bool fresnel = reflectance(cos_theta, refraction_ratio) > u01(rng);
    if (cannot_refract||fresnel){
        //pathSegment.color *= glm::vec3(1.0f)-reflectance(cos_theta, refraction_ratio);
        pathSegment.ray.origin=getPointOnRay(pathSegment.ray, intersection.t);
        pathSegment.ray.direction=glm::reflect(glm::normalize(pathSegment.ray.direction),intersection.surfaceNormal);
    }
    else{
        //pathSegment.color *= glm::vec3(1.0f)-reflectance(cos_theta, refraction_ratio);
        pathSegment.ray.origin=getPointOnRay(pathSegment.ray, intersection.t+0.0002f);
        pathSegment.ray.direction =glm::refract(glm::normalize(pathSegment.ray.direction), intersection.surfaceNormal, refraction_ratio);
        
    }
    pathSegment.remainingBounces--; 
}

__host__ __device__
void specularScatter(PathSegment & pathSegment,
        ShadeableIntersection& intersection,
        glm::vec3 &materialColor,float exponent,thrust::default_random_engine &rng, Geom & light,
    float & lightsize){
    thrust::uniform_real_distribution<float> u01(0, 1);
    float r2=u01(rng);
    float cos=glm::dot(-glm::normalize(pathSegment.ray.direction), intersection.surfaceNormal);
    pathSegment.color *= materialColor;
    pathSegment.remainingBounces--;
    pathSegment.ray.origin=getPointOnRay(pathSegment.ray, intersection.t);
    /*if(r2<0.5f){
        glm::vec3 dist;
        float pdf;
        sampleLight(dist,light,lightsize,rng,pdf,pathSegment.ray.origin);

        pathSegment.ray.direction=glm::normalize(dist-pathSegment.ray.origin);
        //pathSegment.color*=(exponent+1)*pow(glm::abs(glm::dot(pathSegment.ray.direction, intersection.surfaceNormal)),exponent)/TWO_PI;
        pathSegment.remainingBounces=1;
    }else{*/
        glm::vec3 reflected=glm::reflect(pathSegment.ray.direction,intersection.surfaceNormal);
        pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(reflected,rng,exponent));
    //}
    //glass apperance at top
    //bool fresnel = reflectance(cos_theta, refraction_ratio) > u01(rng);
    
    if(dot(pathSegment.ray.direction,intersection.surfaceNormal)<0){
        pathSegment.remainingBounces=0;
        pathSegment.color=glm::vec3(0.0f);
        return;
    }
    
}

__host__ __device__
void blinnScatter(PathSegment & pathSegment,
        ShadeableIntersection& intersection,
        glm::vec3 &materialColor,float exponent,thrust::default_random_engine &rng, Geom & light,
    float & lightsize){
    
    glm::vec3 h =glm::normalize(calculateRandomDirectionInHemisphere(intersection.surfaceNormal,rng,exponent));
    glm::vec3 wo=glm::reflect(pathSegment.ray.direction,h);
    
    glm::vec3 color = materialColor + ((float)pow(1.0f - glm::dot(h, wo), 5.0f)) * (glm::vec3(1.0f) - materialColor);
    float bsdf=(exponent+2.0f)/(2.0f*(2.0f-pow(2.0f,-exponent/2.0f)));
    float pdf= (exponent + 1.0f) /(4.0f*glm::dot(h,wo));
    pathSegment.color *= materialColor;
        //color*bsdf/pdf;
    pathSegment.remainingBounces--;
    pathSegment.ray.origin=getPointOnRay(pathSegment.ray, intersection.t);
    pathSegment.ray.direction=wo;
    
    if(dot(pathSegment.ray.direction,intersection.surfaceNormal)<0){
        pathSegment.remainingBounces=0;
        pathSegment.color=glm::vec3(0.0f);
        return;
    }
    
}

__host__ __device__
void blinnMicScatter(PathSegment & pathSegment,
        ShadeableIntersection& intersection,
        glm::vec3 &materialColor,float exponent,thrust::default_random_engine &rng, Geom & light,
    float & lightsize){

    glm::vec3 h =glm::normalize(calculateRandomDirectionInHemisphere(intersection.surfaceNormal,rng,exponent));
    glm::vec3 wo=glm::reflect(pathSegment.ray.direction,h);

    glm::vec3 color=materialColor+ ((float)pow(1.0f - glm::dot(h, wo), 5))*(glm::vec3(1.0f)-materialColor);
    float D=(exponent+1.0f)/TWO_PI*pow(glm::dot(intersection.surfaceNormal,h),exponent);
    float G=computeG(-glm::normalize(pathSegment.ray.direction),intersection.surfaceNormal,exponent)*computeG(wo,intersection.surfaceNormal,exponent);
    float bsdf=D*G/(4*glm::dot(-glm::normalize(pathSegment.ray.direction),intersection.surfaceNormal));
    float pdf=(exponent+1)*pow(glm::dot(intersection.surfaceNormal,h),exponent)/(4.0f*TWO_PI*glm::dot(h,wo));
    pathSegment.color *= color*bsdf/pdf;
    pathSegment.remainingBounces--;
    pathSegment.ray.origin=getPointOnRay(pathSegment.ray, intersection.t);
    
    pathSegment.ray.direction=wo;
    
    if(dot(pathSegment.ray.direction,intersection.surfaceNormal)<0){
        pathSegment.remainingBounces=0;
        pathSegment.color=glm::vec3(0.0f);
        return;
    }
    
}

__host__ __device__
void diffuseMicScatter(PathSegment & pathSegment,
        ShadeableIntersection& intersection,
        glm::vec3 &materialColor,thrust::default_random_engine &rng, Geom & light,
    float & lightsize){

    float exponent=1.0f;
    glm::vec3 h =glm::normalize(calculateRandomDirectionInHemisphere(intersection.surfaceNormal,rng,exponent));
    glm::vec3 wo=glm::reflect(pathSegment.ray.direction,h);

    glm::vec3 color=materialColor+ ((float)pow(1.0f - glm::dot(h, wo), 5))*(glm::vec3(1.0f)-materialColor);
    float D=(exponent+1.0f)/TWO_PI*pow(glm::dot(intersection.surfaceNormal,h),exponent);
    float G=computeG(-glm::normalize(pathSegment.ray.direction),intersection.surfaceNormal,exponent)*computeG(wo,intersection.surfaceNormal,exponent);
    float bsdf=D*G/(4*glm::dot(-glm::normalize(pathSegment.ray.direction),intersection.surfaceNormal));
    float pdf=(exponent+1)*pow(glm::dot(intersection.surfaceNormal,h),exponent)/(4.0f*TWO_PI*glm::dot(h,wo));
    pathSegment.color *= color*bsdf/pdf;
    pathSegment.remainingBounces--;
    pathSegment.ray.origin=getPointOnRay(pathSegment.ray, intersection.t);
    
    pathSegment.ray.direction=wo;
    
    if(dot(pathSegment.ray.direction,intersection.surfaceNormal)<0){
        pathSegment.remainingBounces=0;
        pathSegment.color=glm::vec3(0.0f);
        return;
    }
    
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
    PathSegment& pathSegment,
    ShadeableIntersection& intersection,
    Material& m,
    thrust::default_random_engine& rng,
    glm::vec3* textPixel,
    glm::vec3 back,
    Geom & light,
    float & lightsize,
    int shading) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    if(glm::dot(-glm::normalize(pathSegment.ray.direction), intersection.surfaceNormal)<=0.0f){
        pathSegment.color *= back;
        pathSegment.remainingBounces=0;
        return;
    }

    glm::vec3 materialColor=m.color;
    if(m.dimg!=-1){
        int x=(int)(intersection.uv.x*m.dwidth);
        int y=(int)((1.0f-intersection.uv.y)*m.dheight);
        int imgidx= y*m.dwidth+x+m.dimgidx;
        materialColor=textPixel[imgidx];
    }
    
    if(m.nimg!=-1){
        int x=(int)(intersection.uv.x*m.nwidth);
        int y=(int)((1.0f-intersection.uv.y)*m.nheight);
        int imgidx= y*m.nwidth+x+m.nimgidx;
        glm::vec3 Bump=textPixel[imgidx];
        Bump=glm::normalize(2.0f*Bump-glm::vec3(1.0f));
        intersection.surfaceNormal=glm::normalize(Bump[0]*intersection.dpdu+Bump[1]*intersection.dpdv+Bump[2]*intersection.surfaceNormal);
        //glm::normalize(Bump[0]*intersection.dpdu+Bump[1]*intersection.dpdv+Bump[2]*intersection.surfaceNormal);
    }

    if (m.emittance > 0.0f) {
        pathSegment.color *= (materialColor * m.emittance);
        pathSegment.remainingBounces=0;
    }else{
        float r1=u01(rng);
        if(r1<m.hasRefractive){
            refractScatter(pathSegment,intersection,materialColor,m.indexOfRefraction,rng);
        }else if(r1<m.hasReflective+m.hasRefractive){
            if(shading==0)
                specularScatter(pathSegment,intersection,m.specular.color,m.specular.exponent,rng,light,lightsize);
            else if(shading==1)
                blinnScatter(pathSegment,intersection,m.specular.color,m.specular.exponent,rng,light,lightsize);
            else
                blinnMicScatter(pathSegment,intersection,m.specular.color,m.specular.exponent,rng,light,lightsize);
        }else{
            diffuseScatter(pathSegment,intersection,materialColor,rng,light,lightsize);
        } 
    }
}
