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

#include "ShadowMap.h"

#include "details/Camera.h"
#include "details/Engine.h"
#include "details/Scene.h"

#include <filament/Engine.h>
#include <filament/LightManager.h>

#include <utils/Entity.h>

#include <math/mat4.h>
#include <math/vec3.h>
#include <math/vec4.h>

#include <gtest/gtest.h>

#include <memory>

using namespace filament;
using namespace filament::math;

class StableShadowMapTest : public testing::Test {
protected:
    void SetUp() override {
        mEngine = downcast(Engine::create(Engine::Backend::NOOP));
        mLight = mEngine->getEntityManager().create();
        mCameraEntity = mEngine->getEntityManager().create();
        mCamera = mEngine->createCamera(mCameraEntity);
        mOptions.stable = true;
        mOptions.shadowFar = 100.0f;
        LightManager::Builder(LightManager::Type::DIRECTIONAL)
                .direction({ 0, -1, 0 })
                .castShadows(true)
                .shadowOptions(mOptions)
                .build(*mEngine, mLight);
        mLightData.push_back();
        mLightData.elementAt<FScene::LIGHT_ENTITY>(0) = mLight;
        mLightData.elementAt<FScene::SHADOW_DIRECTION>(0) = { 0, -1, 0 };
        mLightData.elementAt<FScene::SHADOW_REF>(0) = {};
        mShadowMap = std::make_unique<ShadowMap>(*mEngine);
        mShadowMap->initialize(0, ShadowMap::ShadowLightType::DIRECTIONAL, 0, 0,
                mLight, &mOptions);
    }

    void TearDown() override {
        mShadowMap->terminate(*mEngine);
        mShadowMap.reset();
        mEngine->destroy(mLight);
        mEngine->destroyCameraComponent(mCameraEntity);
        mEngine->getEntityManager().destroy(mLight);
        mEngine->getEntityManager().destroy(mCameraEntity);
        Engine* engine = mEngine;
        Engine::destroy(&engine);
    }

    ShadowMap::ShaderParameters update(float near, float far, mat4 const& model = {}) {
        auto& lm = mEngine->getLightManager();
        lm.setShadowOptions(lm.getInstance(mLight), mOptions);
        mCamera->Camera::setProjection(60.0, 1.0, near, far);
        mCamera->setModelMatrix(model);
        CameraInfo const camera{ *mCamera };
        ShadowMap::ShadowMapInfo const info{
                .atlasDimension = 1024,
                .textureDimension = 1024,
                .shadowDimension = 1022,
        };
        ShadowMap::SceneInfo scene;
        scene.lsCastersNearFar = { 1000, -1000 };
        scene.lsReceiversNearFar = { 1000, -1000 };
        scene.wsShadowCastersVolume = { float3{-1000}, float3{1000} };
        scene.wsShadowReceiversVolume = scene.wsShadowCastersVolume;
        scene.visibleLayers = 0xff;
        auto const result = mShadowMap->updateDirectional(*mEngine, mLightData, 0,
                camera, info, scene, false);
        EXPECT_TRUE(mShadowMap->hasVisibleShadows());
        return result;
    }

    FEngine* mEngine = nullptr;
    FCamera* mCamera = nullptr;
    utils::Entity mLight;
    utils::Entity mCameraEntity;
    LightManager::ShadowOptions mOptions;
    FScene::LightSoa mLightData;
    std::unique_ptr<ShadowMap> mShadowMap;
};

TEST_F(StableShadowMapTest, ShadowFarPreservesCascadeResolution) {
    // The manager has already applied shadowFar and installed each cascade's projection.
    auto const nearCascade = update(0.1f, 10.0f);
    auto const farCascade = update(10.0f, 100.0f);
    EXPECT_LT(nearCascade.texelSizeAtOneMeterWs.x, farCascade.texelSizeAtOneMeterWs.x * 0.2f);
    EXPECT_LT(nearCascade.texelSizeAtOneMeterWs.y, farCascade.texelSizeAtOneMeterWs.y * 0.2f);
}

TEST_F(StableShadowMapTest, ShadowFarMatchesAlreadyClippedProjection) {
    auto const limited = update(0.1f, 20.0f);
    mOptions.shadowFar = 0;
    auto const reference = update(0.1f, 20.0f);
    EXPECT_FLOAT_EQ(limited.texelSizeAtOneMeterWs.x, reference.texelSizeAtOneMeterWs.x);
    EXPECT_FLOAT_EQ(limited.texelSizeAtOneMeterWs.y, reference.texelSizeAtOneMeterWs.y);
    for (size_t column = 0; column < 4; ++column) {
        EXPECT_FLOAT_EQ(limited.lightSpace[column].x, reference.lightSpace[column].x);
        EXPECT_FLOAT_EQ(limited.lightSpace[column].y, reference.lightSpace[column].y);
    }
}

TEST_F(StableShadowMapTest, CameraTranslationIsAppliedOnce) {
    // A camera away from the rendering origin occurs with grid origins or no recentering.
    float3 const offset{ 10, 0, 10 };
    float3 const receiver{ 0, 0, -5 };
    auto const original = update(0.1f, 20.0f);
    auto const moved = update(0.1f, 20.0f, mat4::translation(double3(offset)));
    float3 const uv0 = mat4f::project(original.lightSpace, receiver);
    float3 const uv1 = mat4f::project(moved.lightSpace, receiver + offset);
    // Stable maps snap in two-texel increments, so allow one snap interval.
    EXPECT_NEAR(uv0.x, uv1.x, 2.0f / 1024.0f);
    EXPECT_NEAR(uv0.y, uv1.y, 2.0f / 1024.0f);
}
