/*
 * Copyright (C) 2024 The Android Open Source Project
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

#include <backend/DriverEnums.h>
#include <backend/Handle.h>
#include <backend/platforms/VulkanPlatform.h>

#include <bluevk/BlueVK.h>

namespace test {

using namespace filament;
using namespace filament::backend;
using namespace bluevk;

// Test importing an externally-created VkImage as a render target
TEST_F(BackendTest, ImportVkImageAsRenderTarget) {
    // Skip if not Vulkan backend
    if (sBackend != Backend::VULKAN) {
        GTEST_SKIP() << "Test only applicable to Vulkan backend";
    }

    auto& api = getDriverApi();
    auto* vulkanPlatform = static_cast<VulkanPlatform*>(getPlatform());

    VkDevice device = vulkanPlatform->getDevice();
    VkPhysicalDevice physicalDevice = vulkanPlatform->getPhysicalDevice();

    constexpr uint32_t kWidth = 512;
    constexpr uint32_t kHeight = 512;

    // Create an external VkImage
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent = { kWidth, kHeight, 1 };
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage externalImage;
    VkResult result = vkCreateImage(device, &imageInfo, nullptr, &externalImage);
    ASSERT_EQ(result, VK_SUCCESS);

    // Allocate and bind memory
    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, externalImage, &memReqs);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    uint32_t memTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((memReqs.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            memTypeIndex = i;
            break;
        }
    }
    ASSERT_NE(memTypeIndex, UINT32_MAX);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = memTypeIndex;

    VkDeviceMemory imageMemory;
    result = vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory);
    ASSERT_EQ(result, VK_SUCCESS);

    result = vkBindImageMemory(device, externalImage, imageMemory, 0);
    ASSERT_EQ(result, VK_SUCCESS);

    // Import into Filament
    auto usage = TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLEABLE;
    Handle<HwTexture> texture = api.importTexture(
            reinterpret_cast<intptr_t>(externalImage),
            SamplerType::SAMPLER_2D,
            1,                          // levels
            TextureFormat::RGBA8,
            1,                          // samples
            kWidth, kHeight, 1,         // dimensions
            usage);

    // Create render target using imported texture
    Handle<HwRenderTarget> renderTarget = api.createRenderTarget(
            TargetBufferFlags::COLOR0,
            kWidth, kHeight, 1, 0,
            {{ texture }},              // color attachments
            {},                         // depth attachment
            {});                        // stencil attachment

    // Render to the imported texture
    RenderPassParams params = {};
    params.viewport = { 0, 0, kWidth, kHeight };
    params.flags.clear = TargetBufferFlags::COLOR;
    params.clearColor = { 1.0f, 0.0f, 0.0f, 1.0f };  // Red
    params.flags.discardStart = TargetBufferFlags::ALL;
    params.flags.discardEnd = TargetBufferFlags::NONE;

    auto swapChain = createSwapChain();
    api.makeCurrent(swapChain, swapChain);
    api.beginFrame(0, 0, 0);

    api.beginRenderPass(renderTarget, params);
    api.endRenderPass();

    api.commit(swapChain);
    api.endFrame(0);

    flushAndWait();

    // Cleanup Filament resources
    api.destroyRenderTarget(renderTarget);
    api.destroyTexture(texture);

    executeCommands();

    api.destroySwapChain(swapChain);

    flushAndWait();

    // Cleanup Vulkan resources (Filament should NOT have freed these)
    vkDestroyImage(device, externalImage, nullptr);
    vkFreeMemory(device, imageMemory, nullptr);
}

} // namespace test
