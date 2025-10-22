#version 450

// доп. 1: Принимаем интерполированный цвет из vertex shader
layout (location = 0) in vec3 frag_color;

layout (location = 0) out vec4 final_color;

void main() {
    // доп. 1: Используем интерполированный цвет вместо uniform
    final_color = vec4(frag_color, 1.0f);
}
