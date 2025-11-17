#version 450


//входные данные фрагмента
//приходят из вертекс шейдера
layout (location = 0) in vec3 f_position;//позиция
layout (location = 1) in vec3 f_normal;//нормаль
layout (location = 2) in vec2 f_uv;//текстура
layout (location = 3) in vec3 f_view_pos;//позиция камеры

//выходные данные
layout (location = 0) out vec4 final_color; // цвет пикселя


//данные каждого объекта
layout (binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float shininess;
    vec3 specular_color;
};

//направленный свет
struct DirectionalLight {
    vec3 direction;
    float _pad0;
    vec3 ambient;
    float _pad1;
    vec3 diffuse;
    float _pad2;
    vec3 specular;
    float _pad3;
};

//точечный
struct PointLight {
    vec3 position;
    float _pad0;
    vec3 ambient;
    float _pad1;
    vec3 diffuse;
    float _pad2;
    vec3 specular;
    float constant;
    float linear;
    float quadratic;
    float _pad3;
    float _pad4;
};

//прожектор
struct SpotLight {
    vec3 position;
    float _pad0;
    vec3 direction;
    float _pad1;
    vec3 ambient;
    float _pad2;
    vec3 diffuse;
    float _pad3;
    vec3 specular;
    float cutOff;
    float outerCutOff;
    float constant;
    float linear;
    float quadratic;
    float _pad4;
    float _pad5;
};

//глобальные параметры освещения
layout (binding = 2, std140) uniform LightingUniforms {
    DirectionalLight dirLight;
    int numPointLights;
    int numSpotLights;
};

//динамические массивы источников света
layout (std430, binding = 3) readonly buffer PointLights {
    PointLight lights[];
} pointLights;

layout (std430, binding = 4) readonly buffer SpotLights {
    SpotLight lights[];
} spotLights;

layout (binding = 5) uniform sampler2D texture_sampler;

// Направленный свет (Блинн-Фонг)
vec3 calcDirectionalLight(DirectionalLight light, vec3 normal, vec3 viewDir) {
    vec3 lightDir = normalize(-light.direction);

    // (n·ω) - косинус угла между нормалью и светом
    float n_dot_l = max(dot(normal, lightDir), 0.0);

    vec3 halfwayDir = normalize(lightDir + viewDir);
    float n_dot_h = max(dot(normal, halfwayDir), 0.0);
    float spec_multiplier = pow(n_dot_h, shininess);


    // L_V = [ρ_d(n·l) + ρ_s(n·h)^n](n·l)E_l


    vec3 bracket = albedo_color * n_dot_l +           // ρ_d(n·l)
                   specular_color * spec_multiplier;   // ρ_s(n·h)^n

    // Умножить на (n·l)E_l
    vec3 result = bracket * n_dot_l * light.diffuse;

    // Добавить ambient (L_a)
    result += light.ambient * albedo_color;

    return result;
}

// Точечный источник
vec3 calcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
    // Вектор от фрагмента к свету
    vec3 lightDir = light.position - fragPos;

    // Расстояние в квадрате
    float distanceSqr = max(dot(lightDir, lightDir), 0.00001);
    lightDir = normalize(lightDir);

    //затухание
    float attenuation = 1.0 / distanceSqr;

    float diff = max(dot(normal, lightDir), 0.0);

    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);

    vec3 ambient = light.ambient * albedo_color;
    vec3 diffuse = light.diffuse * diff * albedo_color;
    vec3 specular = light.specular * spec * specular_color;

    return (ambient + diffuse + specular) * attenuation;
}

// Прожектор с нормализацией направления В ШЕЙДЕРЕ
vec3 calcSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
    vec3 lightDir = light.position - fragPos;
    float distanceSqr = max(dot(lightDir, lightDir), 0.00001);
    lightDir = normalize(lightDir);

    //защита от нулевого вектора
    vec3 spotDirection = light.direction;
    float dirLength = length(spotDirection);
    if (dirLength < 0.001) {
        spotDirection = vec3(0.0, -1.0, 0.0); // Дефолт вниз
    } else {
        spotDirection = normalize(spotDirection);
    }

    // Проверка конуса прожектора
    float theta = dot(lightDir, -spotDirection);
    float epsilon = max(light.cutOff - light.outerCutOff, 0.001);
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);

    // Затухание по закону обратных квадратов
    float attenuation = 1.0 / distanceSqr;

    // Диффузная составляющая
    float diff = max(dot(normal, lightDir), 0.0);

    // Блинн-Фонг specular
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);

    vec3 ambient = light.ambient * albedo_color;
    vec3 diffuse = light.diffuse * diff * albedo_color;
    vec3 specular = light.specular * spec * specular_color;

    return (ambient + diffuse * intensity + specular * intensity) * attenuation;
}

void main() {
    vec3 norm = normalize(f_normal);
    vec3 viewDir = normalize(f_view_pos - f_position);

    vec3 result = vec3(0.0);

    // Направленный свет
    result += calcDirectionalLight(dirLight, norm, viewDir);

    // Точечные источники
    for (int i = 0; i < numPointLights; i++) {
        result += calcPointLight(pointLights.lights[i], norm, f_position, viewDir);
    }

    // Прожекторы
    for (int i = 0; i < numSpotLights; i++) {
        result += calcSpotLight(spotLights.lights[i], norm, f_position, viewDir);
    }

     if (abs(norm.y) > 0.9 && abs(f_position.y) < 0.5) {
            // Норма почти вертикальная И позиция близко к полу (y < 0.5)
            vec3 texture_color = texture(texture_sampler, f_uv).rgb;
            result *= texture_color;
     }

    final_color = vec4(result, 1.0);
}
