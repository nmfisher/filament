//------------------------------------------------------------------------------
// Shadowing
//------------------------------------------------------------------------------

#if defined(VARIANT_HAS_SHADOWING)
/**
 * Computes the light space position of the specified world space point.
 * The returned point may contain a bias to attempt to eliminate common
 * shadowing artifacts such as "acne". To achieve this, the world space
 * normal at the point must also be passed to this function.
 * Normal bias is not used for VSM.
 */

highp vec4 computeLightSpacePosition(highp vec3 p, const highp vec3 n,
        const highp vec3 dir, const highp vec2 b, highp_mat4 lightFromWorldMatrix) {

    highp vec4 shadowPosition = mulMat4x4Float3(lightFromWorldMatrix, p);
    // VSM and EVSSM pass zero bias. Avoid computing a basis for these projections.
    if (all(equal(b, vec2(0.0)))) {
        return shadowPosition;
    }

    // b already contains world-space texel lengths. The projection's rows are scaled
    // covectors, not unit light axes; using them directly makes bias depend on projection
    // scale and atlas allocation. Remove the perspective divide's contribution first.
    highp mat4x3 rows = mat4x3(transpose(lightFromWorldMatrix));
    highp vec3 du = rows[0] * shadowPosition.w - rows[3] * shadowPosition.x;
    highp vec3 dv = rows[1] * shadowPosition.w - rows[3] * shadowPosition.y;
    highp vec3 dz = rows[2] * shadowPosition.w - rows[3] * shadowPosition.z;

    // Columns of the inverse projection Jacobian point along world-space texels.
    // Their common determinant and w factors disappear on normalization. This also
    // handles sheared LiSPSM projections, for which simply normalizing rows is incorrect.
    highp vec3 texelU = normalize(cross(dv, dz));
    highp vec3 texelV = normalize(cross(dz, du));
    highp vec2 n_L = vec2(dot(n, texelU), dot(n, texelV));
    p += n * dot(abs(n_L), b);

    return mulMat4x4Float3(lightFromWorldMatrix, p);
}

#endif // VARIANT_HAS_SHADOWING
