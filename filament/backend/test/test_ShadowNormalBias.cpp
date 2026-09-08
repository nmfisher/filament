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

TEST_F(BackendTest, ShadowNormalBiasWorldSpace) {
    constexpr uint32_t width = 10;
    std::string fragment = R"(#version 450 core
#extension GL_GOOGLE_cpp_style_line_directive : enable
precision highp float;
precision highp int;
layout(location = 0) out vec4 fragColor;
#define highp_mat4 mat4
#define VARIANT_HAS_SHADOWING
vec4 mulMat4x4Float3(mat4 m, vec3 p) { return m * vec4(p, 1.0); }
)";
    // Compile the production GLSL, not a C++ copy of its arithmetic.
    fragment += reinterpret_cast<char const*>(SHADERS_SURFACE_SHADOWING_GLSL_DATA);
    fragment += R"(

void main() {
    int testCase = int(gl_FragCoord.x);
    bool perspective = testCase >= 4 && testCase <= 7;
    float mapWidth = testCase == 0 ? 10.0 : 100.0;
    float mapHeight = mapWidth * 0.5;
    vec3 p = vec3(3.0, 2.0, -10.0);
    vec3 n = normalize(vec3(1.0, 1.0, 1.0));
    mat4 m = mat4(1.0);
    m[0][0] = 1.0 / mapWidth;
    m[1][1] = 1.0 / mapHeight;
    m[2][2] = -0.01;
    m[3].xy = vec2(0.5);
    vec2 bias = vec2(mapWidth, mapHeight) / 1024.0;
    if (perspective) {
        // Texture-space perspective projection, with a 2x2 footprint at one meter.
        m = mat4(vec4(0.5, 0, 0, 0), vec4(0, 0.5, 0, 0),
                 vec4(-0.5, -0.5, 0.01, -1), vec4(0, 0, 0.1, 0));
        bias = vec2(2.0 * -p.z / 1024.0);
    }
    if (testCase == 5) { n = vec3(0, 0, 1); }
    float expected = dot(abs(n.xy), bias);
    if (testCase == 2 || testCase == 6) {
        // Repack the same map into an atlas; its world-space texels do not change.
        mat4 atlas = mat4(1.0);
        atlas[0][0] = 0.25;
        atlas[1][1] = 0.5;
        atlas[3].xy = vec2(0.625, 0.125);
        m = atlas * m;
    }
    if (testCase == 3) {
        // Rotate light, camera and geometry together without changing the expected bias.
        mat4 rotation = mat4(vec4(0.8, 0, -0.6, 0), vec4(0, 1, 0, 0),
                             vec4(0.6, 0, 0.8, 0), vec4(0, 0, 0, 1));
        p = (rotation * vec4(p, 1)).xyz;
        n = mat3(rotation) * n;
        m = m * transpose(rotation);
    }
    if (testCase >= 8) {
        m[1].w = 0.02;
        m[1].xy += vec2(0.01);
        if (testCase == 9) {
            mat4 atlas = mat4(1);
            atlas[0][0] = 0.25;
            atlas[1][1] = 0.5;
            atlas[3].xy = vec2(0.625, 0.125);
            m = atlas * m;
        }
        // Independent finite differences of the inverse LiSPSM projection give texel axes.
        mat4 worldFromShadow = inverse(m);
        vec4 q = m * vec4(p, 1);
        q /= q.w;
        float h = 0.0001;
        vec4 px0 = worldFromShadow * (q - vec4(h, 0, 0, 0));
        vec4 px1 = worldFromShadow * (q + vec4(h, 0, 0, 0));
        vec4 py0 = worldFromShadow * (q - vec4(0, h, 0, 0));
        vec4 py1 = worldFromShadow * (q + vec4(0, h, 0, 0));
        vec3 axisX = normalize(px1.xyz / px1.w - px0.xyz / px0.w);
        vec3 axisY = normalize(py1.xyz / py1.w - py0.xyz / py0.w);
        expected = dot(abs(vec2(dot(n, axisX), dot(n, axisY))), bias);
    }
    if (testCase == 7) { bias = vec2(0); expected = 0.0; }
    vec4 result = computeLightSpacePosition(p, n, vec3(0, 0, 1), bias, m);
    // Compare in projection space to avoid amplifying inverse-matrix roundoff in an atlas.
    vec4 unbiased = m * vec4(p, 1);
    vec4 projectedNormal = m * vec4(n, 0);
    float actual = dot(result - unbiased, projectedNormal) / dot(projectedNormal, projectedNormal);
    vec4 expectedPosition = m * vec4(p + n * expected, 1);
    float error = length(result - expectedPosition) / length(projectedNormal);
    fragColor = vec4(actual, expected, error, 1);
}

)";
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
        EXPECT_NEAR(pixels[i * 4], pixels[i * 4 + 1], 0.00002f);
        EXPECT_NEAR(pixels[i * 4 + 2], 0.0f, 0.00002f);
    }
}
} // namespace test
