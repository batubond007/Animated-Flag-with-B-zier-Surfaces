#version 460 core

in vec4 color;
in vec2 TexCoord;

out vec4 fragColor;

uniform sampler2D ourTexture;

void main(void)
{
	// Set the color of this fragment to the interpolated color
	// value computed by the rasterizer.

	//fragColor = color;
	//fragColor = vec4(TexCoord.x, TexCoord.y, 0, 1);
	fragColor = texture(ourTexture, TexCoord) * color;
}
