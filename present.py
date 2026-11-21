"""Last step before the screen: scales the frame to the window and fixes the capture colours, in one shader.

Hardware bilinear (GL_LINEAR) only looks at 2x2 pixels: a 1080p or 1440p picture blown up to a bigger
screen looks soft, and a 4K picture shrunk to a smaller screen skips pixels (shimmer). Here:
  - enlarging: Catmull-Rom 4x4 (9 bilinear reads), kept within the 2x2 neighbours so edges do not ring;
  - shrinking: Catmull-Rom widened by the shrink factor (capped at 2:1, at most 8x8 reads), weights normalised;
  - 1:1: the source pixel, unchanged.
The colour matrix turns RGB decoded with BT.601 (what Windows' DirectShow converter uses for YUY2/NV12)
into RGB decoded with BT.709 (what HD consoles send). It is exact: both decodings are linear in Y'CbCr.
"""
import numpy as np
from OpenGL.GL import (
    GL_COMPILE_STATUS, GL_FRAGMENT_SHADER, GL_LINK_STATUS, GL_TRUE, GL_VERTEX_SHADER, glAttachShader,
    glCompileShader, glCreateProgram, glCreateShader, glGetProgramInfoLog, glGetProgramiv, glGetShaderInfoLog,
    glGetShaderiv, glGetUniformLocation, glLinkProgram, glShaderSource, glUniform1i, glUniform2f,
    glUniformMatrix3fv, glUseProgram,
)

VERTEX = """#version 120
void main() { gl_Position = ftransform(); gl_TexCoord[0] = gl_MultiTexCoord0; }
"""

FRAGMENT = """#version 120
uniform sampler2D tex;
uniform vec2 src;     // texture size, pixels
uniform vec2 ratio;   // screen pixels per source pixel, per axis
uniform int mode;     // 0 = hardware bilinear, 1 = Catmull-Rom
uniform mat3 color;

float cr(float x) {
    x = abs(x);
    if (x < 1.0) return (1.5 * x - 2.5) * x * x + 1.0;
    if (x < 2.0) return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
    return 0.0;
}

vec3 px(vec2 p) { return texture2D(tex, p / src).rgb; }  // p = texel centre, in pixels: exact read

vec3 enlarge(vec2 p) {
    vec2 c = floor(p - 0.5) + 0.5;          // centre of the top-left texel of the 2x2 around p
    vec2 f = p - c;
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2;
    vec2 t0 = c - 1.0, t3 = c + 2.0, t12 = c + w2 / w12;  // the middle pair is one bilinear read
    vec3 r = px(vec2(t0.x, t0.y)) * w0.x * w0.y + px(vec2(t12.x, t0.y)) * w12.x * w0.y + px(vec2(t3.x, t0.y)) * w3.x * w0.y
           + px(vec2(t0.x, t12.y)) * w0.x * w12.y + px(t12) * w12.x * w12.y + px(vec2(t3.x, t12.y)) * w3.x * w12.y
           + px(vec2(t0.x, t3.y)) * w0.x * w3.y + px(vec2(t12.x, t3.y)) * w12.x * w3.y + px(vec2(t3.x, t3.y)) * w3.x * w3.y;
    vec3 a = px(c), b = px(c + vec2(1.0, 0.0)), d = px(c + vec2(0.0, 1.0)), e = px(c + 1.0);
    return clamp(r, min(min(a, b), min(d, e)), max(max(a, b), max(d, e)));
}

vec3 shrink(vec2 p) {
    vec2 s = min(1.0 / min(ratio, vec2(1.0)), vec2(2.0));  // kernel stretch, >= 1 (1 = axis not shrunk)
    vec2 first = floor(p - 2.0 * s - 0.5) + 1.5;           // first texel centre inside the kernel
    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    for (int j = 0; j < 8; j++) {
        float y = first.y + float(j);
        float wy = cr((y - p.y) / s.y);
        if (wy == 0.0) continue;
        for (int i = 0; i < 8; i++) {
            float x = first.x + float(i);
            float w = cr((x - p.x) / s.x) * wy;
            if (w == 0.0) continue;
            sum += px(vec2(x, y)) * w;
            wsum += w;
        }
    }
    return sum / wsum;
}

void main() {
    vec2 uv = gl_TexCoord[0].st;
    vec3 rgb;
    if (mode == 0) rgb = texture2D(tex, uv).rgb;
    else if (ratio.x < 0.999 || ratio.y < 0.999) rgb = shrink(uv * src);
    else rgb = enlarge(uv * src);
    gl_FragColor = vec4(clamp(color * rgb, 0.0, 1.0), 1.0);
}
"""


def ycc_matrix(kr, kb):
    """RGB -> (Y', Cb, Cr) for the given luma weights."""
    kg = 1 - kr - kb
    return np.array([[kr, kg, kb],
                     [-kr / (2 * (1 - kb)), -kg / (2 * (1 - kb)), 0.5],
                     [0.5, -kg / (2 * (1 - kr)), -kb / (2 * (1 - kr))]])


BT601, BT709 = ycc_matrix(0.299, 0.114), ycc_matrix(0.2126, 0.0722)
IDENTITY = np.eye(3, dtype=np.float32)
# RGB that Windows decoded as BT.601 -> RGB decoded as BT.709. Range scaling and offsets cancel out.
FIX_601_TO_709 = (np.linalg.inv(BT709) @ BT601).astype(np.float32)


class Presenter:
    def __init__(self):
        self.prog = glCreateProgram()
        for kind, code in ((GL_VERTEX_SHADER, VERTEX), (GL_FRAGMENT_SHADER, FRAGMENT)):
            sh = glCreateShader(kind)
            glShaderSource(sh, code)
            glCompileShader(sh)
            if glGetShaderiv(sh, GL_COMPILE_STATUS) != GL_TRUE:
                raise RuntimeError(glGetShaderInfoLog(sh).decode(errors="replace"))
            glAttachShader(self.prog, sh)
        glLinkProgram(self.prog)
        if glGetProgramiv(self.prog, GL_LINK_STATUS) != GL_TRUE:
            raise RuntimeError(glGetProgramInfoLog(self.prog).decode(errors="replace"))
        self.loc = {n: glGetUniformLocation(self.prog, n) for n in ("tex", "src", "ratio", "mode", "color")}

    def begin(self, src_w, src_h, view_w, view_h, sharp_scaling, fix_709):
        glUseProgram(self.prog)
        glUniform1i(self.loc["tex"], 0)
        glUniform2f(self.loc["src"], src_w, src_h)
        glUniform2f(self.loc["ratio"], view_w / src_w, view_h / src_h)
        glUniform1i(self.loc["mode"], 1 if sharp_scaling else 0)
        m = FIX_601_TO_709 if fix_709 else IDENTITY
        glUniformMatrix3fv(self.loc["color"], 1, GL_TRUE, np.ascontiguousarray(m))  # numpy rows -> GL columns

    @staticmethod
    def end():
        glUseProgram(0)
