#version 450
#extension GL_KHR_vulkan_glsl : enable
#extension GL_ARB_separate_shader_objects : enable

vec2[] positions = vec2[](
	vec2( 0.0, -0.5),
	vec2( 0.5,  0.5),
	vec2(-0.5,  0.5)
);

vec3[] colors = vec3[](
	vec3(1, 0, 0),
	vec3(0, 1, 0),
	vec3(0, 0, 1)
);

layout (location = 0) out vec3 vertColor;

void main()
{
	gl_Position = vec4(positions[gl_VertexIndex], 0, 1);
	vertColor = colors[gl_VertexIndex];
}