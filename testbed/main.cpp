#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>
#include <veekay/graphics.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace {

	constexpr uint32_t max_models = 1024;


	//веришина
	struct Vertex {
		veekay::vec3 position; //точка
		veekay::vec3 normal; // нормаль
		veekay::vec2 uv; // текстурные координаты
	};

	//глобальные данные всей сцены
	struct SceneUniforms {
		veekay::mat4 view_projection; // матрица преобразования 3д-2д
		veekay::vec3 view_pos; // позиция камеры
		float _pad0; //выравнивание
	};

	//каждый объект(куб, плоскость)
	struct ModelUniforms {
		veekay::mat4 model; //трансформация конретного объекта
		veekay::vec3 albedo_color;// цвет
		float shininess; // блеск
		veekay::vec3 specular_color; // цвет блеска
		float _pad0; //выравнивание
	};

	// направленный свет
	struct DirectionalLight {
		veekay::vec3 direction;  // Направление падения света (куда светит)
		float _pad0;             // Выравнивание памяти
		veekay::vec3 ambient;    // Фоновое освещение (всегда присутствует)
		float _pad1;             // Выравнивание памяти
		veekay::vec3 diffuse;    // Диффузный свет (основной цвет света)
		float _pad2;             // Выравнивание памяти
		veekay::vec3 specular;   // Зеркальный свет (блики)
		float _pad3;             // Выравнивание памяти
	};

	//точечный
	struct PointLight {
		veekay::vec3 position;   // Позиция источника в 3D пространстве
		float _pad0;             // Выравнивание памяти
		veekay::vec3 ambient;    // Фоновое освещение от этого источника
		float _pad1;             // Выравнивание памяти
		veekay::vec3 diffuse;    // Диффузный цвет света от источника
		float _pad2;             // Выравнивание памяти
		veekay::vec3 specular;   // Зеркальный цвет света
		float constant;          // Коэффициент затухания (обычно 1.0)
		float linear;            // Линейный коэффициент затухания (обычно 0.09)
		float quadratic;         // Квадратичный коэффициент затухания (обычно 0.032)
		float _pad3;             // Выравнивание памяти
		float _pad4;             // Выравнивание памяти
	};

	// прожектор
	struct SpotLight {
		veekay::vec3 position;   // Позиция прожектора (где находится фонарик)
		float _pad0;             // Выравнивание памяти
		veekay::vec3 direction;  // Направление конуса прожектора (куда светит)
		float _pad1;             // Выравнивание памяти
		veekay::vec3 ambient;    // Фоновое освещение от прожектора
		float _pad2;             // Выравнивание памяти
		veekay::vec3 diffuse;    // Диффузный цвет прожектора
		float _pad3;             // Выравнивание памяти
		veekay::vec3 specular;   // Зеркальный цвет прожектора
		float cutOff;            // cos(внутренний угол) - область полной яркости
		float outerCutOff;       // cos(внешний угол) - граница конуса (soft edge)
		float constant;          // Коэффициент затухания
		float linear;            // Линейный коэффициент затухания
		float quadratic;         // Квадратичный коэффициент затухания
		float _pad4;             // Выравнивание памяти
		float _pad5;             // Выравнивание памяти
	};

	// освещение всей сцены
	struct LightingUniforms {
		DirectionalLight dirLight;  // Один направленный свет (солнце)
		int numPointLights;         // Количество точечных источников (берётся из массива)
		int numSpotLights;          // Количество прожекторов (берётся из массива)
		float _pad0;                // Выравнивание памяти
		float _pad1;                // Выравнивание памяти
	};


	//геометрия объекта
	struct Mesh {
		veekay::graphics::Buffer* vertex_buffer;  // GPU буфер с вершинами (Vertex структуры)
		veekay::graphics::Buffer* index_buffer;   // GPU буфер с индексами (какие вершины = треугольник)
		uint32_t indices;                         // Количество индексов (для vkCmdDrawIndexed)
	};

	//сдвиг, поворот, масштаб
	struct Transform {
		veekay::vec3 position = {};
		veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
		veekay::vec3 rotation = {};

		// NOTE: Model matrix (translation, rotation and scaling)
		veekay::mat4 matrix() const;
	};

	//объекты сцены
	struct Model {
		Mesh mesh;
		Transform transform;
		veekay::vec3 albedo_color;
	};

	//положение и параметры просмотра
	struct Camera {
		constexpr static float default_fov = 60.0f;
		constexpr static float default_near_plane = 0.01f;
		constexpr static float default_far_plane = 100.0f;

		veekay::vec3 position = {};
		veekay::vec3 rotation = {};

		float fov = default_fov;
		float near_plane = default_near_plane;
		float far_plane = default_far_plane;

		// NOTE: View matrix of camera (inverse of a transform)
		veekay::mat4 view() const;

		// NOTE: View and projection composition
		veekay::mat4 view_projection(float aspect_ratio) const;
	};

	// NOTE: Scene objects
	inline namespace {
		Camera camera{
			.position = {0.0f, -0.5f, 5.0f}
		};

		std::vector<Model> models;
	}

	// NOTE: Vulkan objects
	inline namespace {
		VkShaderModule vertex_shader_module;
		VkShaderModule fragment_shader_module;

		VkDescriptorPool descriptor_pool;
		VkDescriptorSetLayout descriptor_set_layout;
		VkDescriptorSet descriptor_set;

		VkPipelineLayout pipeline_layout;
		VkPipeline pipeline;

		veekay::graphics::Buffer* scene_uniforms_buffer;
		veekay::graphics::Buffer* model_uniforms_buffer;

		Mesh plane_mesh;
		Mesh cube_mesh;

		veekay::graphics::Buffer* lighting_uniforms_buffer;
		veekay::graphics::Buffer* point_lights_buffer;
		veekay::graphics::Buffer* spot_lights_buffer;

		// Параметры освещения
		DirectionalLight dir_light;
		std::vector<PointLight> point_lights;
		std::vector<SpotLight> spot_lights;
		veekay::graphics::Texture* floor_texture;

		// Параметры управления камерой через мышь
		float camera_yaw = 90.0f;                // Угол горизонтального поворота (влево-вправо)
		float camera_pitch = 0.0f;                // Угол вертикального поворота (вверх-вниз)
		veekay::vec3 camera_front = {0.0f, 0.0f, 1.0f};  // Вектор "вперёд" (куда смотрит камера)
		veekay::vec3 camera_right = {1.0f, 0.0f, 0.0f};  // Вектор "вправо" (для WASD движения)
		veekay::vec3 camera_up = {0.0f, 1.0f, 0.0f};     // Вектор "вверх"


		veekay::graphics::Texture* missing_texture;
		VkSampler missing_texture_sampler;

		veekay::graphics::Texture* texture;
		VkSampler texture_sampler;
	}

	float toRadians(float degrees) {
		return degrees * float(M_PI) / 180.0f;
	}

	veekay::mat4 Transform::matrix() const {
		auto t = veekay::mat4::translation(position);
		auto s = veekay::mat4::scaling(scale);

		// Вращение по каждой оси
		auto rx = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, rotation.x);
		auto ry = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, rotation.y);
		auto rz = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, rotation.z);

		return t * ry * rx * rz * s;
	}

	veekay::mat4 Camera::view() const {
		veekay::vec3 center = position + camera_front;
		return veekay::mat4::lookAt(position, center, camera_up);
	}

	veekay::mat4 Camera::view_projection(float aspect_ratio) const {
		auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);

		// перемножение для получения look_at
		return view() * projection;
	}

	// NOTE: Loads shader byte code from file
	// NOTE: Your shaders are compiled via CMake with this code too, look it up
	VkShaderModule loadShaderModule(const char* path) {
		std::ifstream file(path, std::ios::binary | std::ios::ate);
		size_t size = file.tellg();
		std::vector<uint32_t> buffer(size / sizeof(uint32_t));
		file.seekg(0);
		file.read(reinterpret_cast<char*>(buffer.data()), size);
		file.close();

		VkShaderModuleCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = size,
			.pCode = buffer.data(),
		};

		VkShaderModule result;
		if (vkCreateShaderModule(veekay::app.vk_device, &
		                         info, nullptr, &result) != VK_SUCCESS) {
			return nullptr;
		}

		return result;
	}

	veekay::graphics::Texture* loadTextureFromFile(VkCommandBuffer cmd, const char* filepath) {
		int width, height, channels;
		unsigned char* image_data = stbi_load(filepath, &width, &height, &channels, 4);

		if (!image_data) {
			throw std::runtime_error(std::string("Failed to load texture: ") + filepath);
		}

		veekay::graphics::Texture* texture = new veekay::graphics::Texture(
			cmd,
			width,
			height,
			VK_FORMAT_B8G8R8A8_UNORM,
			(uint32_t*)image_data
		);

		stbi_image_free(image_data);
		return texture;
	}


	void initialize(VkCommandBuffer cmd) {
    VkDevice& device = veekay::app.vk_device;
    VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

    // загузка шейдеров
    {
        vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
        if (!vertex_shader_module) {
            std::cerr << "Failed to load Vulkan vertex shader from file\n";
            veekay::app.running = false;
            return;
        }

        fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
        if (!fragment_shader_module) {
            std::cerr << "Failed to load Vulkan fragment shader from file\n";
            veekay::app.running = false;
            return;
        }

        VkPipelineShaderStageCreateInfo stage_infos[2];

        // трансформирует вершины
        stage_infos[0] = VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertex_shader_module,
            .pName = "main",
        };

        // NOTE: Fragment shader stage
        stage_infos[1] = VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragment_shader_module,
            .pName = "main",
        };

        // NOTE: How many bytes does a vertex take?
        VkVertexInputBindingDescription buffer_binding{
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };

        // описание полей вершины (какие данные на каких позициях)
        VkVertexInputAttributeDescription attributes[] = {
            { // позиция
                .location = 0, // NOTE: First attribute
                .binding = 0, // NOTE: First vertex buffer
                .format = VK_FORMAT_R32G32B32_SFLOAT, // NOTE: 3-component vector of floats
                .offset = offsetof(Vertex, position), // NOTE: Offset of "position" field in a Vertex struct
            },
            { // нормаль
                .location = 1,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, normal),
            },
            { // текстуры
                .location = 2,
                .binding = 0,
                .format = VK_FORMAT_R32G32_SFLOAT,
                .offset = offsetof(Vertex, uv),
            },
        };

        // объединяем информацию о вершине
        VkPipelineVertexInputStateCreateInfo input_state_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &buffer_binding,
            .vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
            .pVertexAttributeDescriptions = attributes,
        };

        // треугольники, сборка
        VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        };

        // треугольники, отрисовка, только передние грани
        VkPipelineRasterizationStateCreateInfo raster_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .lineWidth = 1.0f,
        };

        // сглаживание
        VkPipelineMultisampleStateCreateInfo sample_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = false,
            .minSampleShading = 1.0f,
        };

        // область рисования
        VkViewport viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(veekay::app.window_width),
            .height = static_cast<float>(veekay::app.window_height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };

        VkRect2D scissor{
            .offset = {0, 0},
            .extent = {veekay::app.window_width, veekay::app.window_height},
        };

        // NOTE: Let rasterizer draw on the entire window
        VkPipelineViewportStateCreateInfo viewport_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .pViewports = &viewport,
            .scissorCount = 1,
            .pScissors = &scissor,
        };

        // NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
        VkPipelineDepthStencilStateCreateInfo depth_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = true,
            .depthWriteEnable = true,
            .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
        };

        // NOTE: Let fragment shader write all the color channels
        VkPipelineColorBlendAttachmentState attachment_info{
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT |
                VK_COLOR_COMPONENT_A_BIT,
        };

        // NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
        VkPipelineColorBlendStateCreateInfo blend_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = false,
            .logicOp = VK_LOGIC_OP_COPY,
            .attachmentCount = 1,
            .pAttachments = &attachment_info
        };

        // память для дескрипторов
        VkDescriptorPoolSize pools[] = {
            {
                .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 16,
            },
            {
                .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .descriptorCount = 8,
            },
            {
                .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 8,
            },
            {
                .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 16, // Добавлено для SSBO
            },
        };

        VkDescriptorPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = sizeof(pools) / sizeof(pools[0]),
            .pPoolSizes = pools,
        };

        if (vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan descriptor pool\n";
            veekay::app.running = false;
            return;
        }

        // NOTE: Descriptor set layout specification
        VkDescriptorSetLayoutBinding bindings[] = {
            // Binding 0: Scene uniforms (V×P, camera pos)
            {
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            // Binding 1: Model uniforms (M matrix, материал - для каждого объекта)
            {
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            // Binding 2: Lighting uniforms (направленный свет, количество источников)
            {
                .binding = 2,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            // Binding 3: Point lights  (массив точечных источников)
            {
                .binding = 3,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            // Binding 4: Spot lights (массив прожекторов)
            {
                .binding = 4,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
            {
                .binding = 5,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
        };

        VkDescriptorSetLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = sizeof(bindings) / sizeof(bindings[0]),
            .pBindings = bindings,
        };

        if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan descriptor set layout\n";
            veekay::app.running = false;
            return;
        }

        // выделение памяти
        VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptor_pool, // из какого пула
            .descriptorSetCount = 1,
            .pSetLayouts = &descriptor_set_layout, // по какой структуре
        };

        if (vkAllocateDescriptorSets(device, &alloc_info, &descriptor_set) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan descriptor set\n";
            veekay::app.running = false;
            return;
        }

        // СОЗДАНИЕ СЕМПЛЕРА
        VkSamplerCreateInfo sampler_info{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .mipLodBias = 0.0f,
            .maxLod = VK_LOD_CLAMP_NONE,
        };

        if (vkCreateSampler(device, &sampler_info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan texture sampler\n";
            veekay::app.running = false;
            return;
        }

        // Буфер для глобальных данных всей сцены (V×P матрица, позиция камеры)
        // Обновляется один раз в начале render pass'а для всех объектов
        scene_uniforms_buffer = new veekay::graphics::Buffer(
            sizeof(SceneUniforms),
            nullptr,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        // Буфер для данных КАЖДОГО объекта (M матрица, цвет, материал)
        // Каждый куб/объект получит свой элемент из этого буфера
        model_uniforms_buffer = new veekay::graphics::Buffer(
            max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
            nullptr,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        // создание буфферов для освещения
        // буффер для параметров освещения
        lighting_uniforms_buffer = new veekay::graphics::Buffer(
            sizeof(LightingUniforms), nullptr,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        // Инициализировать направленный свет
        dir_light = DirectionalLight{
            .direction = {-0.2f, -1.0f, -0.3f},
            .ambient = {0.2f, 0.2f, 0.2f},
            .diffuse = {0.5f, 0.5f, 0.5f},
            .specular = {1.0f, 1.0f, 1.0f}
        };

        // Добавить точечные источники
        point_lights.push_back(PointLight{
            .position = {1.0f, -0.5f, 1.0f},
            .ambient = {0.1f, 0.1f, 0.1f},
            .diffuse = {0.8f, 0.2f, 0.2f},
            .specular = {1.0f, 1.0f, 1.0f},
            .constant = 1.0f,
            .linear = 0.09f,
            .quadratic = 0.032f
        });

        point_lights.push_back(PointLight{
            .position = {-2.0f, -0.5f, 0.0f},
            .ambient = {0.1f, 0.1f, 0.1f},
            .diffuse = {0.2f, 0.8f, 0.2f},
            .specular = {1.0f, 1.0f, 1.0f},
            .constant = 1.0f,
            .linear = 0.09f,
            .quadratic = 0.032f
        });

        // Добавить прожектор
        spot_lights.push_back(SpotLight{
            .position = {0.0f, -0.5f, 0.0f},
            .direction = {0.0f, -1.0f, 0.0f},
            .ambient = {0.1f, 0.1f, 0.1f},
            .diffuse = {1.0f, 1.0f, 1.0f},
            .specular = {1.0f, 1.0f, 1.0f},
            .cutOff = std::cos(toRadians(12.5f)),
            .outerCutOff = std::cos(toRadians(17.5f)),
            .constant = 1.0f,
            .linear = 0.09f,
            .quadratic = 0.032f
        });

        // Создать SSBO для точечных источников
        point_lights_buffer = new veekay::graphics::Buffer(
            point_lights.size() * sizeof(PointLight), point_lights.data(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        // Создать SSBO для прожекторов
        spot_lights_buffer = new veekay::graphics::Buffer(
            spot_lights.size() * sizeof(SpotLight), spot_lights.data(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        // Загрузить текстуру пола
        floor_texture = loadTextureFromFile(cmd, "/home/mart/CLionProjects/cg_lab_2/assets/1.jpg");

        // NOTE: This texture and sampler is used when texture could not be loaded
        uint32_t pixels[] = {
            0xff000000, 0xffff00ff,
            0xffff00ff, 0xff000000,
        };
        missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
            VK_FORMAT_B8G8R8A8_UNORM,
            pixels);

        // Информация о дескрипторах буферов
        VkDescriptorBufferInfo buffer_infos[] = {
            {
                .buffer = scene_uniforms_buffer->buffer,
                .offset = 0,
                .range = sizeof(SceneUniforms),
            },
            {
                .buffer = model_uniforms_buffer->buffer,
                .offset = 0,
                .range = sizeof(ModelUniforms),
            },
            {
                .buffer = lighting_uniforms_buffer->buffer,
                .offset = 0,
                .range = sizeof(LightingUniforms)
            },
            {
                .buffer = point_lights_buffer->buffer,
                .offset = 0,
                .range = point_lights.size() * sizeof(PointLight)
            },
            {
                .buffer = spot_lights_buffer->buffer,
                .offset = 0,
                .range = spot_lights.size() * sizeof(SpotLight)
            },
        };

        // ИНФОРМАЦИЯ О ТЕКСТУРЕ (с валидным семплером, создан выше)
        VkDescriptorImageInfo texture_info{
            .sampler = missing_texture_sampler,
            .imageView = floor_texture->view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };

        // приявязка буфферов к биндингам
        VkWriteDescriptorSet write_infos[] = {
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptor_set,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &buffer_infos[0],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptor_set,
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .pBufferInfo = &buffer_infos[1],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptor_set,
                .dstBinding = 2,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &buffer_infos[2],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptor_set,
                .dstBinding = 3,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &buffer_infos[3],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptor_set,
                .dstBinding = 4,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &buffer_infos[4],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptor_set,
                .dstBinding = 5,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &texture_info,
            },
        };

        vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
            write_infos, 0, nullptr);

        // NOTE: Declare external data sources, only push constants this time
        VkPipelineLayoutCreateInfo layout_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptor_set_layout,
        };

        // NOTE: Create pipeline layout
        if (vkCreatePipelineLayout(device, &layout_create_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan pipeline layout\n";
            veekay::app.running = false;
            return;
        }

        // Описание всех этапов конвейера
        VkGraphicsPipelineCreateInfo pipeline_info{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = stage_infos,
            .pVertexInputState = &input_state_info,
            .pInputAssemblyState = &assembly_state_info,
            .pViewportState = &viewport_info,
            .pRasterizationState = &raster_info,
            .pMultisampleState = &sample_info,
            .pDepthStencilState = &depth_info,
            .pColorBlendState = &blend_info,
            .layout = pipeline_layout,
            .renderPass = veekay::app.vk_render_pass,
        };

        // NOTE: Create graphics pipeline
        if (vkCreateGraphicsPipelines(device, nullptr, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan pipeline\n";
            veekay::app.running = false;
            return;
        }
    }

    // NOTE: Plane mesh initialization
    // (v0)------(v1)
    // | \     |
    // |  `--, |
    // |      \|
    // (v3)------(v2)
    // сетка плоскости, пол сцены
    {
        std::vector<Vertex> vertices = {
            {{-5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
            // position: (-5, 0, 5) = левый, на полу, передний
            // normal: (0, -1, 0) = вниз (для освещения сверху)
            // uv: (0, 0) = верхний-левый угол текстуры
            {{5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
            {{5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
            {{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
        };

        // индексы двух треугольников
        std::vector<uint32_t> indices = {
            0, 1, 2, 2, 3, 0
        };

        // ГПУ буффер для вершин
        plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
            vertices.size() * sizeof(Vertex), vertices.data(),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

        // для индексов
        plane_mesh.index_buffer = new veekay::graphics::Buffer(
            indices.size() * sizeof(uint32_t), indices.data(),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

        plane_mesh.indices = uint32_t(indices.size());
    }

    // NOTE: Cube mesh initialization
    {
        std::vector<Vertex> vertices = {
            {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
            {{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
            {{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
            {{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},
            {{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
            {{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
            {{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
            {{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
            {{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
            {{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
            {{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
            {{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
            {{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
            {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
            {{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
            {{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
            {{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
            {{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
            {{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
            {{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
            {{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
            {{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
            {{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
            {{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
        };

        // индексы
        std::vector<uint32_t> indices = {
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4,
            8, 9, 10, 10, 11, 8,
            12, 13, 14, 14, 15, 12,
            16, 17, 18, 18, 19, 16,
            20, 21, 22, 22, 23, 20,
        };

        // ГПУ буфферы для вершин и индексов
        cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
            vertices.size() * sizeof(Vertex), vertices.data(),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

        cube_mesh.index_buffer = new veekay::graphics::Buffer(
            indices.size() * sizeof(uint32_t), indices.data(),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

        cube_mesh.indices = uint32_t(indices.size());
    }

    // Добавляем модели на сцену
    // пол
    models.emplace_back(Model{
        .mesh = plane_mesh,
        .transform = Transform{},
        .albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f}
    });

    // красный куб
    models.emplace_back(Model{
        .mesh = cube_mesh,
        .transform = Transform{
            .position = {-2.0f, -0.5f, -1.5f},
        },
        .albedo_color = veekay::vec3{1.0f, 0.0f, 0.0f}
    });

    // зеленый куб
    models.emplace_back(Model{
        .mesh = cube_mesh,
        .transform = Transform{
            .position = {1.5f, -0.5f, -0.5f},
        },
        .albedo_color = veekay::vec3{0.0f, 1.0f, 0.0f}
    });

    // синий куб
    models.emplace_back(Model{
        .mesh = cube_mesh,
        .transform = Transform{
            .position = {0.0f, -0.5f, 1.0f},
        },
        .albedo_color = veekay::vec3{0.0f, 0.0f, 1.0f}
    });
	}


	// отчистка всех ГПУ ресурсов
	void shutdown() {
		VkDevice& device = veekay::app.vk_device;

		vkDestroySampler(device, missing_texture_sampler, nullptr);
		delete floor_texture;

		delete spot_lights_buffer;
		delete point_lights_buffer;
		delete lighting_uniforms_buffer;

		delete missing_texture;

		delete cube_mesh.index_buffer;
		delete cube_mesh.vertex_buffer;

		delete plane_mesh.index_buffer;
		delete plane_mesh.vertex_buffer;

		delete model_uniforms_buffer;
		delete scene_uniforms_buffer;

		vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
		vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
		vkDestroyShaderModule(device, fragment_shader_module, nullptr);
		vkDestroyShaderModule(device, vertex_shader_module, nullptr);
	}


	//обновление
	void update(double time) {
	    ImGui::Begin("Lighting Controls");

	    ImGui::Text("Camera Position: %.2f, %.2f, %.2f",
	        camera.position.x, camera.position.y, camera.position.z);
	    ImGui::Text("Yaw: %.1f, Pitch: %.1f", camera_yaw, camera_pitch);

		//направленный свет
	    ImGui::Separator();
	    ImGui::Text("Directional Light");
	    ImGui::SliderFloat3("Dir Direction", dir_light.direction.elements, -1.0f, 1.0f);
	    ImGui::ColorEdit3("Dir Ambient", dir_light.ambient.elements);
	    ImGui::ColorEdit3("Dir Diffuse", dir_light.diffuse.elements);
	    ImGui::ColorEdit3("Dir Specular", dir_light.specular.elements);

		//точечный
	    ImGui::Separator();
	    for (size_t i = 0; i < point_lights.size(); ++i) {
	        ImGui::PushID(i);
	        ImGui::Text("Point Light %zu", i);
	        ImGui::SliderFloat3("Position", point_lights[i].position.elements, -5.0f, 5.0f);
	        ImGui::ColorEdit3("Diffuse", point_lights[i].diffuse.elements);
	        ImGui::PopID();
	    }

		// прожекторный
	    ImGui::Separator();
		for (size_t i = 0; i < spot_lights.size(); ++i) {
			ImGui::PushID(100 + i);
			ImGui::Text("Spotlight %zu", i);
			ImGui::SliderFloat3("Position", spot_lights[i].position.elements, -5.0f, 5.0f);

			ImGui::Text("Direction");
			ImGui::SliderFloat("Dir X", &spot_lights[i].direction.x, -1.0f, 1.0f);
			ImGui::SliderFloat("Dir Y", &spot_lights[i].direction.y, -1.0f, 1.0f);
			ImGui::SliderFloat("Dir Z", &spot_lights[i].direction.z, -1.0f, 1.0f);


			// Безопасные углы(градусы)
			float inner_deg = 12.5f;
			float outer_deg = 17.5f;

			// Проверка: если cosine значение валидно, конвертировать обратно в градусы
			// cutOff = cos(angle), поэтому angle = acos(cutOff)
			if (spot_lights[i].cutOff >= -1.0f && spot_lights[i].cutOff <= 1.0f) {
				inner_deg = std::acos(spot_lights[i].cutOff) * 180.0f / M_PI;
			}

			// Проверка для outer angle
			if (spot_lights[i].outerCutOff >= -1.0f && spot_lights[i].outerCutOff <= 1.0f) {
				outer_deg = std::acos(spot_lights[i].outerCutOff) * 180.0f / M_PI;
			}


			// Слайдер для внутреннего угла (от 0.1° до 89°)
			if (ImGui::SliderFloat("Inner Angle", &inner_deg, 0.1f, 89.0f)) {
				float radians = inner_deg * M_PI / 180.0f;
				spot_lights[i].cutOff = std::cos(radians);

				//внутренний угол должен быть МЕНЬШЕ внешнего
				// Если внутренний стал больше внешнего, сдвинуть внешний
				if (spot_lights[i].cutOff < spot_lights[i].outerCutOff) {
					spot_lights[i].outerCutOff = spot_lights[i].cutOff - 0.05f;
				}
			}


			if (ImGui::SliderFloat("Outer Angle", &outer_deg, 0.1f, 89.0f)) {
				float radians = outer_deg * M_PI / 180.0f;
				spot_lights[i].outerCutOff = std::cos(radians);
				//внешний угол должен быть БОЛЬШЕ внутреннего
				// cos убывает, поэтому outerCutOff должен быть > cutOff
				if (spot_lights[i].outerCutOff > spot_lights[i].cutOff) {
					spot_lights[i].cutOff = spot_lights[i].outerCutOff + 0.05f;
				}
			}

			ImGui::PopID();
		}

	    ImGui::End();

	    // Управление камерой
	    if (!ImGui::IsWindowHovered() && !ImGui::IsAnyItemActive()) {
	        using namespace veekay::input;

	        // Управление мышью
	        if (mouse::isButtonDown(mouse::Button::left)) {
	            auto delta = mouse::cursorDelta();
	            camera_yaw += delta.x * 0.1f;
	            camera_pitch += delta.y * 0.1f;

	            if (camera_pitch > 89.0f) camera_pitch = 89.0f;
	            if (camera_pitch < -89.0f) camera_pitch = -89.0f;

	            // Обновить векторы камеры
	        	// Преобразовать сферические координаты (yaw, pitch) в декартовы (camera_front)
	            camera_front.x = std::cos(toRadians(camera_yaw)) * std::cos(toRadians(camera_pitch));
	            camera_front.y = std::sin(toRadians(camera_pitch));
	            camera_front.z = std::sin(toRadians(camera_yaw)) * std::cos(toRadians(camera_pitch));

	        	camera_front = veekay::vec3::normalized(camera_front);
	            camera_right = veekay::vec3::normalized(veekay::vec3::cross(camera_front, {0.0f, 1.0f, 0.0f}));
	            camera_up = veekay::vec3::normalized(veekay::vec3::cross(camera_right, camera_front));
	        }

    		if (keyboard::isKeyDown(keyboard::Key::w))
    			camera.position -= camera_front * 0.1f;

    		if (keyboard::isKeyDown(keyboard::Key::s))
    			camera.position += camera_front * 0.1f;

    		if (keyboard::isKeyDown(keyboard::Key::d))
    			camera.position += camera_right * 0.1f;

    		if (keyboard::isKeyDown(keyboard::Key::a))
    			camera.position -= camera_right * 0.1f;

    		if (keyboard::isKeyDown(keyboard::Key::q))
    			camera.position += camera_up * 0.1f;

    		if (keyboard::isKeyDown(keyboard::Key::z))
    			camera.position -= camera_up * 0.1f;
		}

	    // Обновить uniform буферы
	    float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
	    SceneUniforms scene_uniforms{
	        .view_projection = camera.view_projection(aspect_ratio),
	        .view_pos = camera.position
	    };

	    // Обновить освещение
	    LightingUniforms lighting_uniforms{
	        .dirLight = dir_light,
	        .numPointLights = static_cast<int>(point_lights.size()),
	        .numSpotLights = static_cast<int>(spot_lights.size())
	    };

		// Создать вектор ModelUniforms для каждого объекта
	    std::vector<ModelUniforms> model_uniforms(models.size());
	    for (size_t i = 0, n = models.size(); i < n; ++i) {
	        const Model& model = models[i];
	        ModelUniforms& uniforms = model_uniforms[i];
	        uniforms.model = model.transform.matrix();
	        uniforms.albedo_color = model.albedo_color;
	        uniforms.shininess = 32.0f;  // Параметр блеска
	        uniforms.specular_color = veekay::vec3{0.5f, 0.5f, 0.5f};  // Specular цвет
	    }

		// копирование данных в ГПУ буфферы
	    *(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;
	    *(LightingUniforms*)lighting_uniforms_buffer->mapped_region = lighting_uniforms;
	    std::memcpy(point_lights_buffer->mapped_region, point_lights.data(),
	                point_lights.size() * sizeof(PointLight));
	    std::memcpy(spot_lights_buffer->mapped_region, spot_lights.data(),
	                spot_lights.size() * sizeof(SpotLight));

		//копирование модельных данных
	    const size_t alignment =
	        veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));
	    for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
	        const ModelUniforms& uniforms = model_uniforms[i];
	        char* const pointer = static_cast<char*>(model_uniforms_buffer->mapped_region) + i * alignment;
	        *reinterpret_cast<ModelUniforms*>(pointer) = uniforms;
	    }
	}


	void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
		vkResetCommandBuffer(cmd, 0);

		{ // NOTE: Start recording rendering commands
			VkCommandBufferBeginInfo info{
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
				.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
			};

			vkBeginCommandBuffer(cmd, &info);
		}

		{ // NOTE: Use current swapchain framebuffer and clear it
			VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
			VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

			VkClearValue clear_values[] = {clear_color, clear_depth};

			VkRenderPassBeginInfo info{
				.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
				.renderPass = veekay::app.vk_render_pass,
				.framebuffer = framebuffer,
				.renderArea = {
					.extent = {
						veekay::app.window_width,
						veekay::app.window_height
					},
				},
				.clearValueCount = 2,
				.pClearValues = clear_values,
			};

			vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
		}

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		VkDeviceSize zero_offset = 0;

		VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
		VkBuffer current_index_buffer = VK_NULL_HANDLE;

		const size_t model_uniorms_alignment =
			veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

		for (size_t i = 0, n = models.size(); i < n; ++i) {
			const Model& model = models[i];
			const Mesh& mesh = model.mesh;

			if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
				current_vertex_buffer = mesh.vertex_buffer->buffer;
				vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
			}

			if (current_index_buffer != mesh.index_buffer->buffer) {
				current_index_buffer = mesh.index_buffer->buffer;
				vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
			}

			uint32_t offset = i * model_uniorms_alignment;
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
			                    0, 1, &descriptor_set, 1, &offset);

			vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
		}

		vkCmdEndRenderPass(cmd);
		vkEndCommandBuffer(cmd);
	}

} // namespace

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}
