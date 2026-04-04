import { S as ShaderStore } from './index-CxR0Yf_7.js';

// Do not edit.
const name$4 = "prePassDeclaration";
const shader$4 = `#ifdef PREPASS
#extension GL_EXT_draw_buffers : require
layout(location=0) out highp vec4 glFragData[{X}];highp vec4 gl_FragColor;
#ifdef PREPASS_LOCAL_POSITION
varying highp vec3 vPosition;
#endif
#ifdef PREPASS_DEPTH
varying highp vec3 vViewPos;
#endif
#ifdef PREPASS_NORMALIZED_VIEW_DEPTH
varying highp float vNormViewDepth;
#endif
#if defined(PREPASS_VELOCITY) || defined(PREPASS_VELOCITY_LINEAR)
varying highp vec4 vCurrentPosition;varying highp vec4 vPreviousPosition;
#endif
#endif
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name$4]) {
    ShaderStore.IncludesShadersStore[name$4] = shader$4;
}

// Do not edit.
const name$3 = "oitDeclaration";
const shader$3 = `#ifdef ORDER_INDEPENDENT_TRANSPARENCY
#extension GL_EXT_draw_buffers : require
layout(location=0) out vec2 depth; 
layout(location=1) out vec4 frontColor;layout(location=2) out vec4 backColor;
#define MAX_DEPTH 99999.0
highp vec4 gl_FragColor;uniform sampler2D oitDepthSampler;uniform sampler2D oitFrontColorSampler;
#endif
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name$3]) {
    ShaderStore.IncludesShadersStore[name$3] = shader$3;
}

// Do not edit.
const name$2 = "decalFragmentDeclaration";
const shader$2 = `#ifdef DECAL
uniform vec4 vDecalInfos;
#endif
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name$2]) {
    ShaderStore.IncludesShadersStore[name$2] = shader$2;
}

// Do not edit.
const name$1 = "depthPrePass";
const shader$1 = `#ifdef DEPTHPREPASS
gl_FragColor=vec4(0.,0.,0.,1.0);return;
#endif
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name$1]) {
    ShaderStore.IncludesShadersStore[name$1] = shader$1;
}

// Do not edit.
const name = "oitFragment";
const shader = `#ifdef ORDER_INDEPENDENT_TRANSPARENCY
float fragDepth=gl_FragCoord.z; 
#ifdef ORDER_INDEPENDENT_TRANSPARENCY_16BITS
uint halfFloat=packHalf2x16(vec2(fragDepth));vec2 full=unpackHalf2x16(halfFloat);fragDepth=full.x;
#endif
ivec2 fragCoord=ivec2(gl_FragCoord.xy);vec2 lastDepth=texelFetch(oitDepthSampler,fragCoord,0).rg;vec4 lastFrontColor=texelFetch(oitFrontColorSampler,fragCoord,0);depth.rg=vec2(-MAX_DEPTH);frontColor=lastFrontColor;backColor=vec4(0.0);
#ifdef USE_REVERSE_DEPTHBUFFER
float furthestDepth=-lastDepth.x;float nearestDepth=lastDepth.y;
#else
float nearestDepth=-lastDepth.x;float furthestDepth=lastDepth.y;
#endif
float alphaMultiplier=1.0-lastFrontColor.a;
#ifdef USE_REVERSE_DEPTHBUFFER
if (fragDepth>nearestDepth || fragDepth<furthestDepth) {
#else
if (fragDepth<nearestDepth || fragDepth>furthestDepth) {
#endif
return;}
#ifdef USE_REVERSE_DEPTHBUFFER
if (fragDepth<nearestDepth && fragDepth>furthestDepth) {
#else
if (fragDepth>nearestDepth && fragDepth<furthestDepth) {
#endif
depth.rg=vec2(-fragDepth,fragDepth);return;}
#endif
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name]) {
    ShaderStore.IncludesShadersStore[name] = shader;
}
//# sourceMappingURL=oitFragment-mEmMQoBe.js.map
