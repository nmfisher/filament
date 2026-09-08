/*
 * Copyright (C) 2026 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "BackendTest.h"
#include "Shader.h"
#include "TrianglePrimitive.h"

#include "generated/shaders.h"

#include <backend/DriverEnums.h>
#include <backend/PixelBufferDescriptor.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>
#include <utility>

namespace test {
using namespace filament::backend;

TEST_F(BackendTest, ShadowTexelFootprintWorldSpace) {
    constexpr uint32_t width = 6;
    std::string fragment = R"(#version 450 core
precision highp float;
precision highp int;
layout(location = 0) out vec4 fragColor;
#define highp_mat4 mat4
#define VARIANT_HAS_SHADOWING
vec4 mulMat4x4Float3(mat4 m, vec3 p) { return m * vec4(p, 1.0); }
)";
    // Use both the production helper and its production call-site arguments.
    std::string const source(reinterpret_cast<char const*>(SHADERS_SURFACE_SHADOWING_FS_DATA));
    auto const start = source.find("highp vec2 computeTexelSizeInWorldSpace(");
    auto const end = source.find("float chebyshevUpperBound", start);
    ASSERT_NE(start, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    fragment += source.substr(start, end - start);
    auto const call = source.find("computeTexelSizeInWorldSpace(", end);
    auto const callEnd = source.find(';', call);
    ASSERT_NE(call, std::string::npos);
    ASSERT_NE(callEnd, std::string::npos);
    std::string const productionCall = source.substr(call, callEnd - call);
    fragment += R"(

struct Shadow { mat4 lightFromWorldMatrix; };
struct ShadowUniforms { Shadow shadows[1]; };
ShadowUniforms shadowUniforms;

vec3 projectShadow(mat4 m, vec3 p) {
    vec4 q = m * vec4(p, 1);
    // VSM and EVSSM use linear light-space depth, without dividing z by w.
    return vec3(q.xy / q.w, q.z);
}

void main() {
    int testCase = int(gl_FragCoord.x);
    mat4 m = mat4(1);
    m[0][0] = 0.01;
    m[1][1] = 0.02;
    m[2][2] = 0.01;
    m[3].xyz = vec3(0.5);
    vec3 p = vec3(7, 10, -20);
    if (testCase >= 2) {
        // LiSPSM: w depends on light-space y, including the UV translation times w.
        m[1].w = 0.02;
        m[1].xy += vec2(0.01);
    }
    if (testCase == 1 || testCase == 3) {
        mat4 atlas = mat4(1);
        atlas[0][0] = 0.25;
        atlas[1][1] = 0.5;
        atlas[3].xy = vec2(0.625, 0.125);
        m = atlas * m;
    }
    if (testCase == 4) {
        mat4 flip = mat4(1);
        flip[1][1] = -1.0;
        flip[3].y = 1.0;
        m = flip * m;
    }
    if (testCase == 5) {
        mat4 rotation = mat4(vec4(0.8, 0, -0.6, 0), vec4(0, 1, 0, 0),
                             vec4(0.6, 0, 0.8, 0), vec4(0, 0, 0, 1));
        p = (rotation * vec4(p, 1)).xyz;
        m = m * transpose(rotation);
    }
    int index = 0;
    shadowUniforms.shadows[index].lightFromWorldMatrix = m;
    vec4 shadowPosition = m * vec4(p, 1);
    vec3 position = vec3(shadowPosition.xy / shadowPosition.w, shadowPosition.z);
    vec2 texelSize = vec2(1.0 / 1024.0);
    vec2 actual = PRODUCTION_CALL;

    // Independent numerical differentiation of world -> (u, v, linear depth).
    float h = 0.05;
    vec3 dx = vec3(h, 0, 0), dy = vec3(0, h, 0), dz = vec3(0, 0, h);
    mat3 jacobian = mat3(projectShadow(m, p + dx) - projectShadow(m, p - dx),
                         projectShadow(m, p + dy) - projectShadow(m, p - dy),
                         projectShadow(m, p + dz) - projectShadow(m, p - dz)) / (2.0 * h);
    mat3 worldFromShadow = inverse(jacobian);
    vec2 expected = vec2(length(worldFromShadow[0]), length(worldFromShadow[1])) * texelSize;
    fragColor = vec4(actual.x, expected.x, actual.y / expected.y, 1);
}

)";
    auto const marker = fragment.find("PRODUCTION_CALL");
    ASSERT_NE(marker, std::string::npos);
    fragment.replace(marker, sizeof("PRODUCTION_CALL") - 1, productionCall);
    auto& api = getDriverApi();
    auto swapChain = addCleanup(createSwapChain());
    api.makeCurrent(swapChain, swapChain);
    Shader shader(api, *mCleanup, ShaderConfig{
            .vertexShader = R"(#version 450 core
layout(location = 0) in vec4 mesh_position;
void main() { gl_Position = vec4((mesh_position.xy + 0.5) * 5.0, 0.0, 1.0); }
)",
            .fragmentShader = fragment,
    });
    TrianglePrimitive triangle(api);
    auto texture = addCleanup(api.createTexture(SamplerType::SAMPLER_2D, 1,
            TextureFormat::RGBA32F, 1, width, 1, 1, TextureUsage::COLOR_ATTACHMENT));
    auto target = addCleanup(api.createRenderTarget(TargetBufferFlags::COLOR,
            width, 1, 1, 0, {{texture}}, {}, {}));
    PipelineState state = getColorWritePipelineState();
    shader.addProgramToPipelineState(state);
    state.primitiveType = PrimitiveType::TRIANGLES;
    state.vertexBufferInfo = triangle.getVertexBufferInfo();
    RenderPassParams params{};
    params.viewport = {0, 0, width, 1};
    params.flags.clear = TargetBufferFlags::COLOR;
    params.flags.discardStart = TargetBufferFlags::ALL;
    params.flags.discardEnd = TargetBufferFlags::NONE;
    api.beginFrame(0, 0, 0);
    api.beginRenderPass(target, params);
    api.bindPipeline(state);
    api.bindRenderPrimitive(triangle.getRenderPrimitive());
    api.draw2(0, 3, 1);
    api.endRenderPass();
    std::array<float, width * 4> pixels{};
    bool callbackCalled = false;
    PixelBufferDescriptor descriptor(pixels.data(), sizeof(pixels), PixelDataFormat::RGBA,
            PixelDataType::FLOAT, [](void*, size_t, void* user) {
                *static_cast<bool*>(user) = true;
            }, &callbackCalled);
    api.readPixels(target, 0, 0, width, 1, std::move(descriptor));
    api.commit(swapChain);
    api.endFrame(0);
    flushAndWait();
    ASSERT_TRUE(callbackCalled);
    for (uint32_t i = 0; i < width; ++i) {
        SCOPED_TRACE(i);
        EXPECT_FLOAT_EQ(pixels[i * 4 + 3], 1.0f); // Also proves the shader ran.
        EXPECT_GT(pixels[i * 4 + 1], 0.0f);
        EXPECT_NEAR(pixels[i * 4], pixels[i * 4 + 1], pixels[i * 4 + 1] * 0.005f);
        EXPECT_NEAR(pixels[i * 4 + 2], 1.0f, 0.005f);
    }
}
} // namespace test
