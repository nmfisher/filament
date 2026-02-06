/*
 * Copyright (C) 2025 The Android Open Source Project
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

/**
 * This test demonstrates a bug where transparent objects are picked even when
 * transparent picking is disabled (the default behavior).
 *
 * The bug is in RenderPass.cpp: when rendering the depth pass for picking with
 * FILTER_TRANSLUCENT_OBJECTS enabled, translucent objects have their depthWrite
 * disabled but the command is NOT cancelled. This means they still write to the
 * picking buffer via colorWrite.
 *
 * Expected behavior: With setTransparentPickingEnabled(false) (the default),
 * picking should NOT return transparent objects.
 *
 * Actual behavior: Transparent objects ARE returned in pick results.
 */

#include <gtest/gtest.h>

#include <filamat/MaterialBuilder.h>

#include <filament/Camera.h>
#include <filament/Engine.h>
#include <filament/IndexBuffer.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderableManager.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/Skybox.h>
#include <filament/SwapChain.h>
#include <filament/VertexBuffer.h>
#include <filament/View.h>
#include <filament/Viewport.h>

#include <utils/EntityManager.h>

#include <chrono>
#include <condition_variable>
#include <mutex>

using namespace filament;
using namespace utils;

namespace {

// Simple triangle vertices (a quad made of 2 triangles, centered at origin in XY plane)
struct Vertex {
    filament::math::float3 position;
};

static const Vertex QUAD_VERTICES[4] = {
    {{ -1.0f, -1.0f, 0.0f }},
    {{  1.0f, -1.0f, 0.0f }},
    {{  1.0f,  1.0f, 0.0f }},
    {{ -1.0f,  1.0f, 0.0f }},
};

static constexpr uint16_t QUAD_INDICES[6] = { 0, 1, 2, 0, 2, 3 };

class TransparentPickingTest : public testing::Test {
protected:
    Engine* mEngine = nullptr;
    SwapChain* mSwapChain = nullptr;
    Renderer* mRenderer = nullptr;
    View* mView = nullptr;
    Scene* mScene = nullptr;
    Camera* mCamera = nullptr;
    Entity mCameraEntity;
    Skybox* mSkybox = nullptr;

    VertexBuffer* mVertexBuffer = nullptr;
    IndexBuffer* mIndexBuffer = nullptr;
    Material* mTransparentMaterial = nullptr;
    Entity mTransparentRenderable;

    void SetUp() override {
        // Use OpenGL backend for picking support
        mEngine = Engine::create(Engine::Backend::OPENGL);
        if (!mEngine) {
            GTEST_SKIP() << "OpenGL backend not available";
        }

        mSwapChain = mEngine->createSwapChain(64, 64);
        mRenderer = mEngine->createRenderer();
        mScene = mEngine->createScene();

        mCameraEntity = EntityManager::get().create();
        mCamera = mEngine->createCamera(mCameraEntity);
        // Position camera looking down Z axis at origin
        mCamera->lookAt({0, 0, 5}, {0, 0, 0});

        mView = mEngine->createView();
        mView->setViewport({0, 0, 64, 64});
        mView->setScene(mScene);
        mView->setCamera(mCamera);
        mView->setPostProcessingEnabled(false);

        // Explicitly disable transparent picking (this is also the default)
        mView->setTransparentPickingEnabled(false);

        mSkybox = Skybox::Builder().color({0.0f, 0.0f, 0.0f, 1.0f}).build(*mEngine);
        mScene->setSkybox(mSkybox);

        // Create vertex buffer
        mVertexBuffer = VertexBuffer::Builder()
                .vertexCount(4)
                .bufferCount(1)
                .attribute(VertexAttribute::POSITION, 0, VertexBuffer::AttributeType::FLOAT3, 0, sizeof(Vertex))
                .build(*mEngine);
        mVertexBuffer->setBufferAt(*mEngine, 0,
                VertexBuffer::BufferDescriptor(QUAD_VERTICES, sizeof(QUAD_VERTICES), nullptr));

        // Create index buffer
        mIndexBuffer = IndexBuffer::Builder()
                .indexCount(6)
                .bufferType(IndexBuffer::IndexType::USHORT)
                .build(*mEngine);
        mIndexBuffer->setBuffer(*mEngine,
                IndexBuffer::BufferDescriptor(QUAD_INDICES, sizeof(QUAD_INDICES), nullptr));

        // Create a transparent material using MaterialBuilder
        filamat::MaterialBuilder builder;
        builder.init();
        builder.name("TransparentMaterial");
        builder.shading(filamat::MaterialBuilder::Shading::UNLIT);
        // Set blending to TRANSPARENT (or FADE) - this makes the material translucent
        builder.blending(filamat::MaterialBuilder::BlendingMode::TRANSPARENT);
        builder.material(R"(
            void material(inout MaterialInputs material) {
                prepareMaterial(material);
                material.baseColor = vec4(1.0, 0.0, 0.0, 0.5);  // Semi-transparent red
            }
        )");

        filamat::Package package = builder.build(mEngine->getJobSystem());
        ASSERT_TRUE(package.isValid()) << "Failed to build transparent material";

        mTransparentMaterial = Material::Builder()
                .package(package.getData(), package.getSize())
                .build(*mEngine);
        ASSERT_NE(mTransparentMaterial, nullptr);

        // Create renderable entity with transparent material
        mTransparentRenderable = EntityManager::get().create();
        RenderableManager::Builder(1)
                .boundingBox({{ -1, -1, -1 }, { 1, 1, 1 }})
                .material(0, mTransparentMaterial->getDefaultInstance())
                .geometry(0, RenderableManager::PrimitiveType::TRIANGLES,
                         mVertexBuffer, mIndexBuffer, 0, 6)
                .culling(false)
                .receiveShadows(false)
                .castShadows(false)
                .build(*mEngine, mTransparentRenderable);

        mScene->addEntity(mTransparentRenderable);
    }

    void TearDown() override {
        if (mEngine) {
            mScene->remove(mTransparentRenderable);
            mEngine->destroy(mTransparentRenderable);
            mEngine->destroy(mTransparentMaterial);
            mEngine->destroy(mIndexBuffer);
            mEngine->destroy(mVertexBuffer);
            mEngine->destroyCameraComponent(mCameraEntity);
            EntityManager::get().destroy(mCameraEntity);
            mEngine->destroy(mScene);
            mEngine->destroy(mView);
            mEngine->destroy(mSkybox);
            mEngine->destroy(mRenderer);
            mEngine->destroy(mSwapChain);
            Engine::destroy(&mEngine);
        }
    }

    // Render a few frames and return pick result
    View::PickingQueryResult performPick(uint32_t x, uint32_t y) {
        std::mutex mutex;
        std::condition_variable cv;
        bool pickComplete = false;
        View::PickingQueryResult pickResult{};

        // Set up pick callback
        mView->pick(x, y, [&](View::PickingQueryResult const& result) {
            std::lock_guard<std::mutex> lock(mutex);
            pickResult = result;
            pickComplete = true;
            cv.notify_one();
        });

        // Render frames until pick completes (picking has latency of a few frames)
        for (int i = 0; i < 10 && !pickComplete; i++) {
            if (mRenderer->beginFrame(mSwapChain)) {
                mRenderer->render(mView);
                mRenderer->endFrame();
            }
            mEngine->flushAndWait();

            std::unique_lock<std::mutex> lock(mutex);
            if (pickComplete) break;
        }

        // Wait for pick result
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait_for(lock, std::chrono::seconds(5), [&] { return pickComplete; });
        }

        EXPECT_TRUE(pickComplete) << "Pick callback was never called";
        return pickResult;
    }
};

/**
 * Test that transparent objects are NOT picked when transparent picking is disabled.
 *
 * This test currently FAILS, demonstrating the bug.
 *
 * The expected behavior is that with setTransparentPickingEnabled(false),
 * picking at the center of the viewport (where the transparent quad is)
 * should return an empty/null entity.
 *
 * The actual behavior is that the transparent object IS picked despite
 * transparent picking being disabled.
 */
TEST_F(TransparentPickingTest, TransparentObjectNotPickedWhenDisabled) {
    // Pick at center of viewport where the transparent quad is visible
    View::PickingQueryResult result = performPick(32, 32);

    // With transparent picking disabled, the transparent renderable should NOT be picked
    // BUG: Currently this assertion FAILS because the transparent object IS picked
    EXPECT_FALSE(result.renderable)
        << "Transparent object was picked even though transparent picking is disabled. "
        << "Entity ID: " << result.renderable.getId();

    // Alternative check: if an entity was picked, it should NOT be our transparent one
    if (result.renderable) {
        EXPECT_NE(result.renderable, mTransparentRenderable)
            << "The transparent renderable was incorrectly picked when transparent picking is disabled";
    }
}

/**
 * Test that transparent objects ARE picked when transparent picking is enabled.
 * This test verifies that the picking mechanism works correctly for transparent objects
 * when it's supposed to (i.e., when explicitly enabled).
 */
TEST_F(TransparentPickingTest, TransparentObjectPickedWhenEnabled) {
    // Enable transparent picking
    mView->setTransparentPickingEnabled(true);

    // Pick at center of viewport where the transparent quad is visible
    View::PickingQueryResult result = performPick(32, 32);

    // With transparent picking enabled, the transparent renderable SHOULD be picked
    EXPECT_TRUE(result.renderable) << "No entity was picked";
    EXPECT_EQ(result.renderable, mTransparentRenderable)
        << "Expected to pick the transparent renderable when transparent picking is enabled";
}

} // anonymous namespace
