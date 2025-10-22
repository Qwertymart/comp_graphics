#version 450

// доп. 1: Добавлен атрибут цвета
layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_color;

// доп.1: Цвет передается во fragment shader для интерполяции
layout (location = 0) out vec3 frag_color;

// Push constants больше не содержат цвет
layout (push_constant, std430) uniform ShaderConstants {
    mat4 projection;
    mat4 transform;
};

void main() {
    vec4 point = vec4(v_position, 1.0f);
    vec4 transformed = transform * point;
    vec4 projected = projection * transformed;

    gl_Position = projected;

    // доп. 1: Передаем цвет для интерполяции между вершинами
    frag_color = v_color;
}
