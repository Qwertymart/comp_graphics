#version 450


//входные данные
layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

//в фрагментный шейдер
layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;
layout (location = 3) out vec3 f_view_pos;

//сцена
layout (binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec3 view_pos;
};

layout (binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float shininess;
    vec3 specular_color;
};

void main() {
    //трансформация позиции
    vec4 position = model * vec4(v_position, 1.0f);
    //трансформации нормали
    mat3 normal_matrix = transpose(inverse(mat3(model)));
    vec3 normal = normal_matrix * v_normal;

    //финальная позиция в экранных коорд
    gl_Position = view_projection * position;

    //передача в фрагментный шейдер
    f_position = position.xyz;
    f_normal = normal;
    f_uv = v_uv;
    f_view_pos = view_pos;
}
