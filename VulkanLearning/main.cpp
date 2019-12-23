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

#ifdef NDEBUG
#	define IS_DEBUG_BUILD false
#else
#	define IS_DEBUG_BUILD true
#endif

#define ARRAY_SIZE(x) (sizeof(x)/sizeof(0[x]))

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
	static constexpr int WIDTH = 800;
	static constexpr int HEIGHT = 600;
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
		//TODO: Query for presenation support
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
		auto queueFamilies = findQueueFamilies(device);
		return queueFamilies.graphicsFamily.has_value()
			&& queueFamilies.presentFamily.has_value();
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

	void mainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
		}
	}

	void cleanup()
	{
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