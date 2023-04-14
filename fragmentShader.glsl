#version 330 core

out vec4 color;

float c2f(int color) {
	if(color < 0) return 0;
	if(color > 255) return 1;
	return color/255.0;
}

void main() {
	color = vec4(c2f(129), c2f(212), c2f(250), 1.0);
}
