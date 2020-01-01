#include <volk.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <functional>
#include <cstdlib>
#include <memory>
#include <vector>
#include <cstring>
#include <cassert>
#include <optional>
#include <unordered_set>
#include <algorithm>
#include <fstream>
#include <string>
#include <cerrno>

#ifdef NDEBUG
#	define IS_DEBUG_BUILD false
#else
#	define IS_DEBUG_BUILD true
#endif

#define ARRAY_SIZE(x) (sizeof(x)/sizeof(0[x]))

std::string readFile(const char* const filename)
{
	std::ifstream in(filename, std::ios::in | std::ios::binary);
	if (in)
	{
		std::string contents;
		in.seekg(0, std::ios::end);
		contents.resize(in.tellg());
		in.seekg(0, std::ios::beg);
		in.read(&contents[0], contents.size());
		in.close();
		return(contents);
	}
	throw std::runtime_error("failed to open file (error code " + std::to_string(errno) + ")");
}

class VulkanApplication
{
public:
	virtual void run() = 0;

	virtual ~VulkanApplication()
	{
	}
};

class HelloTriangleApp : public VulkanApplication
{
private: //static
	static constexpr unsigned WIDTH = 800;
	static constexpr unsigned HEIGHT = 600;
	static constexpr auto TITLE = "Learning Vulkan";
	static constexpr const char* validationLayers[] = {
		"VK_LAYER_KHRONOS_validation"
	};
	static constexpr bool enableValidationLayers = IS_DEBUG_BUILD;

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData
	) 
	{
		std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

		if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
		{
			__debugbreak();
		}

		return VK_FALSE;
	}

	struct QueueFamilyIndices
	{
		std::optional<uint32_t> graphicsFamily;
		std::optional<uint32_t> computeFamily;
		std::optional<uint32_t> transferFamily;
		std::optional<uint32_t> sparseBindingFamily;
		std::optional<uint32_t> presentFamily;
	};

	static inline const std::vector<const char*> deviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};

	struct SwapChainSupportDetails 
	{
		VkSurfaceCapabilitiesKHR capabilities;
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;
	};

	static constexpr inline const char* const VERTEX_SHADER_PATH = "vert.spv";
	static constexpr inline const char* const FRAGMENT_SHADER_PATH = "frag.spv";
private:
	GLFWwindow* window;
	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkSurfaceKHR surface;
	VkPhysicalDevice physicalDevice;
	QueueFamilyIndices queueFamilyIndices;
	VkDevice device;
	VkQueue graphicsQueue;
	VkQueue presentQueue;
	VkSwapchainKHR swapChain;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImage> swapChainImages;
	std::vector<VkImageView> swapChainImageViews;
	VkRenderPass renderPass;
	VkPipelineLayout pipelineLayout;
	VkPipeline graphicsPipeline;
	std::vector<VkFramebuffer> framebuffers;
public:
	virtual void run() override
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	void initWindow()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		window = glfwCreateWindow(WIDTH, HEIGHT, TITLE, nullptr, nullptr);
	}

	void initVulkan()
	{
		loadGlobalFunctions();
		createInstance();
		loadInstanceFunctions();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		cacheQueueFamilyIndices();
		createLogicalDevice();
		retrieveQueueHandles();
		loadDeviceFunctions();
		createSwapChain();
		retrieveSwapChainImageHandles();
		createSwapChainImageViews();
		createRenderPass();
		createGraphicsPipeline();
		createFramebuffers();
	}

	void loadGlobalFunctions()
	{
		if (volkInitialize() != VK_SUCCESS)
			throw std::runtime_error("couldn't load global vulkan functions");
	}

	void loadInstanceFunctions()
	{
		volkLoadInstance(instance);
	}

	void createInstance()
	{
		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Learning Vulkan";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_1;

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		//Extensions
		auto extensions = getRequiredExtensions();
		auto supportedExtensions = getSupportedExtensions();

		if (!checkExtensionSupport(extensions, supportedExtensions))
		{
			throw std::runtime_error("some requested extensions are not supported");
		}

		createInfo.enabledExtensionCount = extensions.size();
		createInfo.ppEnabledExtensionNames = extensions.data();

		auto availableLayers = getSupportedLayers();

		//Validation layers
		if (enableValidationLayers && !checkValidationLayerSupport(availableLayers))
		{
			throw std::runtime_error("some requested validation layers are not supported");
		}

		VkDebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo = createDebugMessengerCreateInfo();
		if (enableValidationLayers)
		{
			createInfo.enabledLayerCount = ARRAY_SIZE(validationLayers);
			createInfo.ppEnabledLayerNames = validationLayers;
			
			createInfo.pNext = &debugMessengerCreateInfo;
		}
		else
		{
			createInfo.enabledLayerCount = 0;
		}

		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create VkInstance");
		}
	}

	std::vector<const char*> getRequiredExtensions()
	{
		uint32_t glfwExtensionCount = 0;
		const char** const glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		if (enableValidationLayers)
		{
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME); // Needed for custom validation layer message callback
		}

		return extensions;
	}

	std::vector<VkExtensionProperties> getSupportedExtensions()
	{
		uint32_t extensionCount = 0;
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
		
		std::vector<VkExtensionProperties> extensions(extensionCount);
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

		return extensions;
	}

	std::vector<VkLayerProperties> getSupportedLayers()
	{
		uint32_t layerCount = 0;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		return availableLayers;
	}

	bool checkValidationLayerSupport(const std::vector<VkLayerProperties>& availableLayers)
	{
		for (const auto& layerName : validationLayers)
		{
			if (!layerNamePresentInLayerVector(layerName, availableLayers))
			{
				return false;
			}
		}

		return true;
	}

	bool checkExtensionSupport(const std::vector<const char*>& extensions, const std::vector<VkExtensionProperties>& availableExtensions)
	{
		for (const auto& extensionName : extensions)
		{
			if (!extensionNamePresentInExtensionVector(extensionName, availableExtensions))
			{
				return false;
			}
		}

		return true;
	}

	bool extensionNamePresentInExtensionVector(const char* extensionName, const std::vector<VkExtensionProperties>& supportedExtensions)
	{
		for (const auto& supportedExtension : supportedExtensions)
		{
			if (cstrEquals(supportedExtension.extensionName, extensionName))
			{
				return true;
			}
		}
		return false;
	}

	bool layerNamePresentInLayerVector(const char* layerName, const std::vector<VkLayerProperties>& supportedLayers)
	{
		for (const auto& supportedLayer : supportedLayers)
		{
			if (cstrEquals(supportedLayer.layerName, layerName))
			{
				return true;
			}
		}
		return false;
	}

	bool cstrEquals(const char* str1, const char* str2)
	{
		return strcmp(str1, str2) == 0;
	}

	void createSurface()
	{
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
		{
			throw std::runtime_error("cannot create window surface");
		}
	}

	void setupDebugMessenger()
	{
		if (!enableValidationLayers)
			return;

		VkDebugUtilsMessengerCreateInfoEXT createInfo = createDebugMessengerCreateInfo();

		if (vkCreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
			throw std::runtime_error("cannot setup debug messenger");
	}

	VkDebugUtilsMessengerCreateInfoEXT createDebugMessengerCreateInfo()
	{
		VkDebugUtilsMessengerCreateInfoEXT createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity =
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType =
			VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = &HelloTriangleApp::debugCallback;
		createInfo.pUserData = static_cast<void*>(this);

		return createInfo;
	}

	void pickPhysicalDevice()
	{
		auto physicalDevices = findPhysicalDevices();
		physicalDevice = pickPhysicalDeviceWithHighestScore(physicalDevices);
	}

	std::vector<VkPhysicalDevice> findPhysicalDevices()
	{
		uint32_t device_count = 0;
		vkEnumeratePhysicalDevices(instance, &device_count, nullptr);

		if (device_count == 0)
		{
			throw std::runtime_error("failed to find gpu supporting vulkan");
		}

		std::vector<VkPhysicalDevice> physicalDevices(device_count);
		vkEnumeratePhysicalDevices(instance, &device_count, physicalDevices.data());
		
		return physicalDevices;
	}

	static constexpr int DEVICE_NON_SUITABLE_SCORE = 0;
	VkPhysicalDevice pickPhysicalDeviceWithHighestScore(const std::vector<VkPhysicalDevice>& physicalDevices)
	{
		int highest_score = DEVICE_NON_SUITABLE_SCORE;
		int best_device_idx = 0;
		for (int i = 0; i < physicalDevices.size(); ++i)
		{
			int score = calculatePhysicalDeviceScore(physicalDevices[i]);
			if (score > highest_score)
			{
				highest_score = score;
				best_device_idx = i;
			}
		}

		if (highest_score == DEVICE_NON_SUITABLE_SCORE)
		{
			throw std::runtime_error("no suitable gpu found");
		}

		return physicalDevices[best_device_idx];
	}

	int calculatePhysicalDeviceScore(VkPhysicalDevice device)
	{
		VkPhysicalDeviceProperties device_properties;
		vkGetPhysicalDeviceProperties(device, &device_properties);

		VkPhysicalDeviceFeatures device_features;
		vkGetPhysicalDeviceFeatures(device, &device_features);

		auto score = DEVICE_NON_SUITABLE_SCORE;
		
		if (!isDeviceSuitable(device, device_properties, device_features))
		{
			return score;
		}
		else
		{
			++score;
		}

		if (device_properties.deviceType & VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
		{
			score += 1000;
		}

		return score; 
	}

	bool isDeviceSuitable(VkPhysicalDevice device, const VkPhysicalDeviceProperties& properties, const VkPhysicalDeviceFeatures& features)
	{
		bool extensionsSupported = checkExtensionSupport(deviceExtensions, getSupportedDeviceExtensions(device));

		bool swapChainAdequate = false;
		if (extensionsSupported)
		{
			SwapChainSupportDetails details = querySwapChainSupport(device);
			swapChainAdequate = !(details.formats.empty() || details.presentModes.empty());
		}

		auto queueFamilies = findQueueFamilies(device);
		return queueFamilies.graphicsFamily.has_value()
			&& queueFamilies.presentFamily.has_value()
			&& extensionsSupported
			&& swapChainAdequate;
	}

	std::vector<VkExtensionProperties> getSupportedDeviceExtensions(VkPhysicalDevice device)
	{
		uint32_t extensionCount = 0;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> extensions(extensionCount);

		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, extensions.data());

		return extensions;
	}

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
	{
		return {
			querySwapChainCapabilities(device),
			querySwapChainFormats(device),
			querySwapChainPresentModes(device)
		};
	}

	VkSurfaceCapabilitiesKHR querySwapChainCapabilities(VkPhysicalDevice device)
	{
		VkSurfaceCapabilitiesKHR result;
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &result);
		return result;
	}

	std::vector<VkSurfaceFormatKHR> querySwapChainFormats(VkPhysicalDevice device)
	{
		uint32_t formatCount = 0;

		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

		std::vector<VkSurfaceFormatKHR> formats(formatCount);

		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, formats.data());

		return formats;
	}

	std::vector<VkPresentModeKHR> querySwapChainPresentModes(VkPhysicalDevice device)
	{
		uint32_t presentModeCount = 0;

		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

		std::vector<VkPresentModeKHR> presentModes(presentModeCount);

		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, presentModes.data());

		return presentModes;
	}

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
	{
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
		
		return {
			findQueueFamilyWithCapability(queueFamilies, VkQueueFlagBits::VK_QUEUE_GRAPHICS_BIT),
			findQueueFamilyWithCapability(queueFamilies, VkQueueFlagBits::VK_QUEUE_COMPUTE_BIT),
			findQueueFamilyWithCapability(queueFamilies, VkQueueFlagBits::VK_QUEUE_TRANSFER_BIT),
			findQueueFamilyWithCapability(queueFamilies, VkQueueFlagBits::VK_QUEUE_SPARSE_BINDING_BIT),
			findPresentationQueueFamily(device, queueFamilies)
		};
	}

	std::optional<uint32_t> findQueueFamilyWithCapability(const std::vector<VkQueueFamilyProperties>& queueFamilies, VkQueueFlagBits capability)
	{
		std::optional<uint32_t> index;

		for (int i = 0; i < queueFamilies.size(); ++i)
		{
			if (queueFamilies[i].queueFlags & capability)
			{
				index = i;
				if ((queueFamilies[i].queueFlags ^ capability) == 0)
				{
					return index;
				}
			}
		}

		return index;
	}

	std::optional<uint32_t> findPresentationQueueFamily(VkPhysicalDevice device, const std::vector<VkQueueFamilyProperties>& queueFamilies)
	{
		for (int i = 0; i < queueFamilies.size(); ++i)
		{
			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
			if (presentSupport)
			{
				return i;
			}
		}

		return std::nullopt;
	}

	void cacheQueueFamilyIndices()
	{
		queueFamilyIndices = findQueueFamilies(physicalDevice);
	}

	void createLogicalDevice()
	{
		VkDeviceCreateInfo deviceCreateInfo{};
		deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

		float queuePriority = 1;
		auto queueCreateInfos = createQueueCreateInfos(&queuePriority);
		deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
		deviceCreateInfo.queueCreateInfoCount = queueCreateInfos.size();

		VkPhysicalDeviceFeatures wantedFeatures{};
		deviceCreateInfo.pEnabledFeatures = &wantedFeatures;

		deviceCreateInfo.enabledExtensionCount = deviceExtensions.size();
		deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

		if (enableValidationLayers)
		{
			deviceCreateInfo.enabledLayerCount = ARRAY_SIZE(validationLayers);
			deviceCreateInfo.ppEnabledLayerNames = validationLayers;
		}
		else
		{
			deviceCreateInfo.enabledLayerCount = 0;
		}

		if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create logical device");
		}
	}

	std::vector<VkDeviceQueueCreateInfo> createQueueCreateInfos(const float* const queuePriority)
	{
		std::unordered_set<uint32_t> uniqueQueueFamilyIndices{
			queueFamilyIndices.graphicsFamily.value(),
			queueFamilyIndices.presentFamily.value()
		};

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		queueCreateInfos.reserve(uniqueQueueFamilyIndices.size());

		for (const auto& queueFamilyIndex : uniqueQueueFamilyIndices)
		{
			VkDeviceQueueCreateInfo queueCreateInfo{};

			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.pQueuePriorities = queuePriority;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.queueFamilyIndex = queueFamilyIndex;

			queueCreateInfos.push_back(queueCreateInfo);
		}

		return queueCreateInfos;
	}

	void retrieveQueueHandles()
	{
		vkGetDeviceQueue(device, queueFamilyIndices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(device, queueFamilyIndices.presentFamily.value(), 0, &presentQueue);
	}

	void loadDeviceFunctions()
	{
		volkLoadDevice(device);
	}

	void createSwapChain()
	{
		auto capabilities = querySwapChainCapabilities(physicalDevice);
		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(querySwapChainFormats(physicalDevice));
		VkPresentModeKHR presentMode = chooseSwapPresentMode(querySwapChainPresentModes(physicalDevice));
		VkExtent2D extent = chooseSwapExtent(capabilities);

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;

		uint32_t imageCount = capabilities.minImageCount + 1;	
		if (capabilities.maxImageCount != 0) // there is an upper limit
		{
			if (imageCount > capabilities.maxImageCount)
			{ 
				imageCount = capabilities.maxImageCount;
			}
		}

		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		uint32_t indices[] = { queueFamilyIndices.graphicsFamily.value(), queueFamilyIndices.presentFamily.value() };
		if (queueFamilyIndices.graphicsFamily != queueFamilyIndices.presentFamily)
		{
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = indices;
		}
		else
		{
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}

		createInfo.preTransform = capabilities.currentTransform;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;

		createInfo.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create swapchain");
		}

		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
	}

	void retrieveSwapChainImageHandles()
	{
		uint32_t imageCount = 0;
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		
		swapChainImages.resize(imageCount);

		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
	}

	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
	{
		for (const auto& format : availableFormats)
		{
			if (format.format == VK_FORMAT_B8G8R8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			{
				return format;
			}
		}

		return availableFormats[0];
	}

	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
	{
		for (const auto& presentMode : availablePresentModes)
		{
			if (presentMode == VkPresentModeKHR::VK_PRESENT_MODE_MAILBOX_KHR)
			{
				return VK_PRESENT_MODE_MAILBOX_KHR; // triple buffering
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR; // guaranteed - double buffering
	}

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
	{
		if (capabilities.currentExtent.width == UINT32_MAX) // the resolution of the surface is not set
		{
			return {
				std::clamp(WIDTH, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
				std::clamp(HEIGHT, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
			};
		}
		else // the resolution of the surface is already set
		{
			return capabilities.currentExtent;
		}
	}

	void createSwapChainImageViews()
	{
		for (const auto& image : swapChainImages)
			swapChainImageViews.push_back(createSwapChainImageView(image));
	}

	VkImageView createSwapChainImageView(VkImage image)
	{
		VkImageViewCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		createInfo.image = image;
		createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		createInfo.format = swapChainImageFormat;
		createInfo.components = { 
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY
		};
		createInfo.subresourceRange = {
			VK_IMAGE_ASPECT_COLOR_BIT,
			0,
			1,
			0,
			1
		};

		VkImageView result;
		if (vkCreateImageView(device, &createInfo, nullptr, &result) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create image view");
		}
		return result;
	}

	void createRenderPass()
	{
		VkAttachmentDescription colorAttachment{};
		colorAttachment.format = swapChainImageFormat;
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		
		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		
		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;

		VkRenderPassCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		createInfo.attachmentCount = 1;
		createInfo.pAttachments = &colorAttachment;
		createInfo.subpassCount = 1;
		createInfo.pSubpasses = &subpass;

		if (vkCreateRenderPass(device, &createInfo, nullptr, &renderPass) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create render pass");
		}
	}

	void createGraphicsPipeline()
	{
		VkShaderModule vertexShader = createShaderModule(readFile(VERTEX_SHADER_PATH));
		VkShaderModule fragmentShader = createShaderModule(readFile(FRAGMENT_SHADER_PATH));

		VkPipelineShaderStageCreateInfo shaderStages[] = {
			createPipelineShaderStageCreateInfo(vertexShader, VK_SHADER_STAGE_VERTEX_BIT),
			createPipelineShaderStageCreateInfo(fragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT)
		};

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexAttributeDescriptionCount = 0;
		vertexInputInfo.vertexBindingDescriptionCount = 0;

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = false;

		VkViewport viewport{};
		viewport.x = 0;
		viewport.y = 0;
		viewport.width = swapChainExtent.width;
		viewport.height = swapChainExtent.height;
		viewport.minDepth = 0.0;
		viewport.maxDepth = 1.0;

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;

		createPipelineLayout();

		VkGraphicsPipelineCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		createInfo.stageCount = 2;
		createInfo.pStages = shaderStages;
		createInfo.pVertexInputState = &vertexInputInfo;
		createInfo.pInputAssemblyState = &inputAssembly;
		createInfo.pViewportState = &viewportState;
		createInfo.pRasterizationState = &rasterizer;
		createInfo.pMultisampleState = &multisampling;
		createInfo.pDepthStencilState = nullptr;
		createInfo.pColorBlendState = &colorBlending;
		createInfo.pDynamicState = nullptr;
		createInfo.layout = pipelineLayout;
		createInfo.renderPass = renderPass;
		createInfo.subpass = 0;
		createInfo.basePipelineHandle = VK_NULL_HANDLE;
		createInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
		{
			vkDestroyShaderModule(device, vertexShader, nullptr);
			vkDestroyShaderModule(device, fragmentShader, nullptr);

			throw std::runtime_error("failed to create graphics pipeline");
		}

		vkDestroyShaderModule(device, vertexShader, nullptr);
		vkDestroyShaderModule(device, fragmentShader, nullptr);
	}

	void createPipelineLayout()
	{
		VkPipelineLayoutCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		createInfo.setLayoutCount = 0;
		createInfo.pushConstantRangeCount = 0;
		
		if (vkCreatePipelineLayout(device, &createInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create pipeline layout");
		}
	}

	VkPipelineShaderStageCreateInfo createPipelineShaderStageCreateInfo(VkShaderModule shader, VkShaderStageFlagBits stageBit)
	{
		VkPipelineShaderStageCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		createInfo.module = shader;
		createInfo.stage = stageBit;
		createInfo.pName = "main";

		return createInfo;
	}

	VkShaderModule createShaderModule(const std::string& code)
	{
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
		createInfo.codeSize = code.size();

		VkShaderModule shader;
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shader) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create shader module");
		}
		return shader;
	}

	void createFramebuffers()
	{
		framebuffers.resize(swapChainImageViews.size(), VK_NULL_HANDLE);
		for (int i = 0; i < framebuffers.size(); ++i)
		{
			VkFramebufferCreateInfo createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			createInfo.renderPass = renderPass;
			createInfo.attachmentCount = 1;
			createInfo.pAttachments = &swapChainImageViews[i];
			createInfo.width = swapChainExtent.width;
			createInfo.height = swapChainExtent.height;
			createInfo.layers = 1;
			
			if (vkCreateFramebuffer(device, &createInfo, nullptr, &framebuffers[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create frambuffer");
			}
		}
	}

	void mainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
		}
	}

	void cleanup()
	{
		for (const auto& framebuffer : framebuffers)
		{
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}
		
		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyRenderPass(device, renderPass, nullptr);
		
		for (const auto imageView : swapChainImageViews)
		{
			vkDestroyImageView(device, imageView, nullptr);
		}
		
		vkDestroySwapchainKHR(device, swapChain, nullptr);

		vkDestroyDevice(device, nullptr);
		
		if (enableValidationLayers)
		{
			vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}

		vkDestroySurfaceKHR(instance, surface, nullptr);

		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);
		glfwTerminate();
	}
};

int main()
{
	try
	{
		std::unique_ptr<VulkanApplication> app = std::make_unique<HelloTriangleApp>();
		app->run();
	}
	catch (const std::exception & e)
	{
		std::cerr << e.what() << '\n';
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}