"""Last step before the screen: scales the frame to the window and fixes the capture colours, in one shader.

Hardware bilinear (GL_LINEAR) only looks at 2x2 pixels: a 1080p or 1440p picture blown up to a bigger
screen looks soft, and a 4K picture shrunk to a smaller screen skips pixels (shimmer). Here:
  - enlarging: Catmull-Rom 4x4 (9 bilinear reads), kept within the 2x2 neighbours so edges do not ring;
  - shrinking: Catmull-Rom widened by the shrink factor (capped at 2:1, at most 8x8 reads), weights normalised;
  - 1:1: the source pixel, unchanged.
The colour matrix turns RGB decoded with BT.601 (what Windows' DirectShow converter uses for YUY2/NV12)
into RGB decoded with BT.709 (what HD consoles send). It is exact: both decodings are linear in Y'CbCr.
FRAGMENT_FX adds light picture effects (debanding, local contrast, vibrance); see tools/check_present.py.
"""
import numpy as np
from OpenGL.GL import (
    GL_COMPILE_STATUS, GL_FRAGMENT_SHADER, GL_LINK_STATUS, GL_TRUE, GL_VERTEX_SHADER, glAttachShader,
    glCompileShader, glCreateProgram, glCreateShader, glGetProgramInfoLog, glGetProgramiv, glGetShaderInfoLog,
    glGetShaderiv, glGetUniformLocation, glLinkProgram, glShaderSource, glUniform1f, glUniform1i, glUniform2f,
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


# The same scaling and colour matrix, then three light picture effects on every displayed frame (real and
# in-between): debanding, local contrast, vibrance. GLSL 1.30 for textureLod (an exact mip level: level 0 for
# the scaling reads, a blurred level for local contrast). If this does not compile, FRAGMENT above is used.
VERTEX_FX = """#version 130
void main() { gl_Position = ftransform(); gl_TexCoord[0] = gl_MultiTexCoord0; }
"""

FRAGMENT_FX = """#version 130
uniform sampler2D tex;
uniform vec2 src;
uniform vec2 ratio;
uniform int mode;
uniform mat3 color;
uniform float deband;  // 0-1, 0 = off
uniform float lcon;    // 0-1, local contrast
uniform float vib;     // 0-1, vibrance
uniform float lod;     // mip level of the local-contrast blur, < 0 when the texture has no mipmaps

float cr(float x) {
    x = abs(x);
    if (x < 1.0) return (1.5 * x - 2.5) * x * x + 1.0;
    if (x < 2.0) return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
    return 0.0;
}

vec3 px(vec2 p) { return textureLod(tex, p / src, 0.0).rgb; }

vec3 enlarge(vec2 p) {
    vec2 c = floor(p - 0.5) + 0.5;
    vec2 f = p - c;
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2;
    vec2 t0 = c - 1.0, t3 = c + 2.0, t12 = c + w2 / w12;
    vec3 r = px(vec2(t0.x, t0.y)) * w0.x * w0.y + px(vec2(t12.x, t0.y)) * w12.x * w0.y + px(vec2(t3.x, t0.y)) * w3.x * w0.y
           + px(vec2(t0.x, t12.y)) * w0.x * w12.y + px(t12) * w12.x * w12.y + px(vec2(t3.x, t12.y)) * w3.x * w12.y
           + px(vec2(t0.x, t3.y)) * w0.x * w3.y + px(vec2(t12.x, t3.y)) * w12.x * w3.y + px(vec2(t3.x, t3.y)) * w3.x * w3.y;
    vec3 a = px(c), b = px(c + vec2(1.0, 0.0)), d = px(c + vec2(0.0, 1.0)), e = px(c + 1.0);
    return clamp(r, min(min(a, b), min(d, e)), max(max(a, b), max(d, e)));
}

vec3 shrink(vec2 p) {
    vec2 s = min(1.0 / min(ratio, vec2(1.0)), vec2(2.0));
    vec2 first = floor(p - 2.0 * s - 0.5) + 1.5;
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

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
vec3 fetch(vec2 uv) { return color * textureLod(tex, uv, 0.0).rgb; }

void main() {
    vec2 uv = gl_TexCoord[0].st;
    vec3 rgb;
    if (mode == 0) rgb = textureLod(tex, uv, 0.0).rgb;
    else if (ratio.x < 0.999 || ratio.y < 0.999) rgb = shrink(uv * src);
    else rgb = enlarge(uv * src);
    rgb = color * rgb;
    vec2 fc = gl_FragCoord.xy;

    // 1. Debanding: four samples on a cross of random angle and radius (8-16 screen pixels). Where they all stay
    // within a few levels of this pixel (a smooth gradient), their average replaces it; an edge, a HUD letter or
    // real texture differs by more and is left alone.
    if (deband > 0.0) {
        float a = hash(fc) * 6.2831853;
        float r = 8.0 + 8.0 * hash(fc + 17.31);
        vec2 to_uv = 1.0 / (ratio * src);
        vec2 u1 = vec2(cos(a), sin(a)) * r * to_uv, u2 = vec2(-sin(a), cos(a)) * r * to_uv;
        vec3 s1 = fetch(uv + u1), s2 = fetch(uv - u1), s3 = fetch(uv + u2), s4 = fetch(uv - u2);
        vec3 avg = (s1 + s2 + s3 + s4) * 0.25;
        float thr = 0.025 * deband;  // 0.4 -> 2.5 levels of 255
        vec3 dm = max(max(abs(s1 - rgb), abs(s2 - rgb)), max(abs(s3 - rgb), abs(s4 - rgb)));
        float far = max(dm.r, max(dm.g, dm.b));
        vec3 da = abs(avg - rgb);
        float near = max(da.r, max(da.g, da.b));
        float w = (1.0 - smoothstep(0.5 * thr, thr, near)) * (1.0 - smoothstep(2.0 * thr, 3.0 * thr, far));
        rgb = mix(rgb, avg, w);
    }

    // 2. Local contrast, on luma: the pixel against a wide blur (a mip level, about 16 screen pixels), faded out in
    // deep shadows and bright highlights. The added relief is capped so strong edges do not get halos.
    if (lcon > 0.0 && lod >= 0.0) {
        float l = luma(rgb);
        float b = luma(color * textureLod(tex, uv, lod).rgb);
        float w = smoothstep(0.03, 0.18, l) * (1.0 - smoothstep(0.72, 0.97, l));
        rgb += lcon * 0.8 * clamp(l - b, -0.06, 0.06) * w;
    }

    // 3. Vibrance: dull colours gain more saturation than vivid ones; skin tones (orange hues, moderate
    // saturation) are kept nearly as they are.
    if (vib > 0.0) {
        float mx = max(rgb.r, max(rgb.g, rgb.b)), mn = min(rgb.r, min(rgb.g, rgb.b));
        float sat = mx - mn;
        float skin = 0.0;
        if (rgb.r >= rgb.g && rgb.g >= rgb.b && sat > 0.02) {
            float h = (rgb.g - rgb.b) / sat;  // 0 = red, 1 = yellow
            skin = smoothstep(0.1, 0.25, h) * (1.0 - smoothstep(0.65, 0.85, h)) * (1.0 - smoothstep(0.35, 0.6, sat));
        }
        float amt = vib * 0.5 * (1.0 - sat) * (1.0 - sat) * (1.0 - 0.85 * skin);
        float l = luma(rgb);
        rgb = mix(vec3(l), rgb, 1.0 + amt);
    }

    // Dither before the 8-bit output, so smoothed gradients do not band again (triangular, +-1 level).
    if (deband > 0.0) rgb += (hash(fc + 0.5) + hash(fc + 3.7) - 1.0) / 255.0;
    gl_FragColor = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
"""

# Default strengths: light, the picture must stay true to the game.
FX_DEFAULTS = {"deband": 0.4, "lcon": 0.25, "vib": 0.25}


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


def build_program(vertex, fragment):
    prog = glCreateProgram()
    for kind, code in ((GL_VERTEX_SHADER, vertex), (GL_FRAGMENT_SHADER, fragment)):
        sh = glCreateShader(kind)
        glShaderSource(sh, code)
        glCompileShader(sh)
        if glGetShaderiv(sh, GL_COMPILE_STATUS) != GL_TRUE:
            raise RuntimeError(glGetShaderInfoLog(sh).decode(errors="replace"))
        glAttachShader(prog, sh)
    glLinkProgram(prog)
    if glGetProgramiv(prog, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(prog).decode(errors="replace"))
    return prog


class Presenter:
    def __init__(self, effects=True):
        """effects: try the shader with the picture effects first; if it does not compile, the plain one."""
        self.fx = False
        self.fx_error = None
        if effects:
            try:
                self.prog = build_program(VERTEX_FX, FRAGMENT_FX)
                self.fx = True
            except Exception as e:
                self.fx_error = str(e).strip()
        if not self.fx:
            self.prog = build_program(VERTEX, FRAGMENT)
        names = ("tex", "src", "ratio", "mode", "color") + (("deband", "lcon", "vib", "lod") if self.fx else ())
        self.loc = {n: glGetUniformLocation(self.prog, n) for n in names}

    def begin(self, src_w, src_h, view_w, view_h, sharp_scaling, fix_709, fx=None, lod=-1.0):
        """fx: {"deband", "lcon", "vib"} strengths, or None for no effect; lod: blur mip level (< 0: none)."""
        glUseProgram(self.prog)
        glUniform1i(self.loc["tex"], 0)
        glUniform2f(self.loc["src"], src_w, src_h)
        glUniform2f(self.loc["ratio"], view_w / src_w, view_h / src_h)
        glUniform1i(self.loc["mode"], 1 if sharp_scaling else 0)
        m = FIX_601_TO_709 if fix_709 else IDENTITY
        glUniformMatrix3fv(self.loc["color"], 1, GL_TRUE, np.ascontiguousarray(m))  # numpy rows -> GL columns
        if self.fx:
            fx = fx or {}
            glUniform1f(self.loc["deband"], float(fx.get("deband", 0.0)))
            glUniform1f(self.loc["lcon"], float(fx.get("lcon", 0.0)))
            glUniform1f(self.loc["vib"], float(fx.get("vib", 0.0)))
            glUniform1f(self.loc["lod"], float(lod))

    @staticmethod
    def blur_lod(src_w, view_w, radius=16.0):
        """Mip level whose texels span about `radius` screen pixels (the local-contrast blur)."""
        return max(0.0, float(np.log2(max(radius * src_w / max(view_w, 1), 1.0))))

    @staticmethod
    def end():
        glUseProgram(0)
