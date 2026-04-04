import { aq as EngineStore, bb as DrawWrapper, h as RandomGUID, bY as Effect, S as ShaderStore, a9 as PushMaterial, x as PrepareDefinesForMisc, y as PrepareDefinesForFrameBoundValues, z as PrepareDefinesForAttributes, K as PrepareAttributesForInstances, X as PrepareUniformsAndSamplersList, Y as AddClipPlaneUniforms, b1 as Camera, a2 as BindClipPlane, a4 as BindFogParameters, a6 as BindLogDepth, e as SerializationHelper, ac as MaterialDefines, J as VertexBuffer, R as RegisterClass, bZ as functions, b_ as HighestCommonFactor, l as Mesh, V as Vector3, m as VertexData, M as Matrix, b$ as SubMesh, c as TmpVectors, L as Logger, c0 as runCoroutineAsync, c1 as createYieldingScheduler, c2 as ImportMeshAsync, aO as ToHalfFloat, c3 as runCoroutineSync, ah as Vector2, b9 as Color4, ao as Quaternion, C as Color3, ai as Vector4, bA as RandomRange, B as BaseTexture, c4 as SPLATFileLoaderMetadata, d as Tools, ax as RegisterSceneLoaderPlugin } from './index-CxR0Yf_7.js';
import { af as __vitePreload } from './index-BvBk1Iap.js';
import { ShaderMaterial } from './shaderMaterial-UCHwgcla.js';
import './clipPlaneFragment-Vm6wRK67.js';
import './logDepthDeclaration-DzD3v8nZ.js';
import './fogFragment-Ek2G8c0v.js';
import './sceneUboDeclaration-CbezEhV0.js';
import './meshUboDeclaration-zsmLsV73.js';
import './clipPlaneVertex-BKsOyoIZ.js';
import './logDepthVertex-DP62dfil.js';
import './helperFunctions-CiFVmLcr.js';
import './clipPlaneFragment-BegocNAa.js';
import './logDepthDeclaration-BkiOZiuv.js';
import './fogFragment-DbmcEy2X.js';
import './sceneUboDeclaration-BjEv3Rnv.js';
import './meshUboDeclaration-CkiQ-QPA.js';
import './helperFunctions-qLR1vvLh.js';
import './clipPlaneVertex-BeUdHTek.js';
import './logDepthVertex-CB3XkllG.js';
import { R as RawTexture } from './rawTexture-Bto4WZO2.js';
import './thinInstanceMesh-Bxr7w30K.js';
import { A as AssetContainer } from './assetContainer-7GPvH2S_.js';
import { Ray } from './ray-BjtzqJE9.js';
import { S as StandardMaterial } from './standardMaterial-BzBQSFg4.js';

class MapMap {
    constructor() {
        this.mm = new Map();
    }
    get(a, b) {
        const m = this.mm.get(a);
        if (m !== undefined) {
            return m.get(b);
        }
        return undefined;
    }
    set(a, b, v) {
        let m = this.mm.get(a);
        if (m === undefined) {
            this.mm.set(a, (m = new Map()));
        }
        m.set(b, v);
    }
}
/**
 * Class that can be used to wrap a base material to generate accurate shadows when using custom vertex/fragment code in the base material
 */
class ShadowDepthWrapper {
    /** Gets the standalone status of the wrapper */
    get standalone() {
        return this._options?.standalone ?? false;
    }
    /** Gets the base material the wrapper is built upon */
    get baseMaterial() {
        return this._baseMaterial;
    }
    /** Gets the doNotInjectCode status of the wrapper */
    get doNotInjectCode() {
        return this._options?.doNotInjectCode ?? false;
    }
    /**
     * Instantiate a new shadow depth wrapper.
     * It works by injecting some specific code in the vertex/fragment shaders of the base material and is used by a shadow generator to
     * generate the shadow depth map. For more information, please refer to the documentation:
     * https://doc.babylonjs.com/features/featuresDeepDive/lights/shadows
     * @param baseMaterial Material to wrap
     * @param scene Define the scene the material belongs to
     * @param options Options used to create the wrapper
     */
    constructor(baseMaterial, scene, options) {
        this._baseMaterial = baseMaterial;
        this._scene = scene ?? EngineStore.LastCreatedScene;
        this._options = options;
        this._subMeshToEffect = new Map();
        this._subMeshToDepthWrapper = new MapMap();
        this._meshes = new Map();
        // Register for onEffectCreated to store the effect of the base material when it is (re)generated. This effect will be used
        // to create the depth effect later on
        this._onEffectCreatedObserver = this._baseMaterial.onEffectCreatedObservable.add((params) => {
            const mesh = params.subMesh?.getMesh();
            if (mesh && !this._meshes.has(mesh)) {
                // Register for mesh onDispose to clean up our internal maps when a mesh is disposed
                this._meshes.set(mesh, mesh.onDisposeObservable.add((mesh) => {
                    const iterator = this._subMeshToEffect.keys();
                    for (let key = iterator.next(); key.done !== true; key = iterator.next()) {
                        const subMesh = key.value;
                        if (subMesh?.getMesh() === mesh) {
                            this._subMeshToEffect.delete(subMesh);
                            this._deleteDepthWrapperEffect(subMesh);
                        }
                    }
                }));
            }
            if (this._subMeshToEffect.get(params.subMesh)?.[0] !== params.effect) {
                this._subMeshToEffect.set(params.subMesh, [params.effect, this._scene.getEngine().currentRenderPassId]);
                this._deleteDepthWrapperEffect(params.subMesh);
            }
        });
    }
    _deleteDepthWrapperEffect(subMesh) {
        const depthWrapperEntries = this._subMeshToDepthWrapper.mm.get(subMesh);
        if (depthWrapperEntries) {
            // find and release the previous depth effect
            depthWrapperEntries.forEach((depthWrapper) => {
                depthWrapper.mainDrawWrapper.effect?.dispose();
            });
            this._subMeshToDepthWrapper.mm.delete(subMesh); // trigger a depth effect recreation
        }
    }
    /**
     * Gets the effect to use to generate the depth map
     * @param subMesh subMesh to get the effect for
     * @param shadowGenerator shadow generator to get the effect for
     * @param passIdForDrawWrapper Id of the pass for which the effect from the draw wrapper must be retrieved from
     * @returns the effect to use to generate the depth map for the subMesh + shadow generator specified
     */
    getEffect(subMesh, shadowGenerator, passIdForDrawWrapper) {
        const entry = this._subMeshToDepthWrapper.mm.get(subMesh)?.get(shadowGenerator);
        if (!entry) {
            return null;
        }
        let drawWrapper = entry.drawWrapper[passIdForDrawWrapper];
        if (!drawWrapper) {
            drawWrapper = entry.drawWrapper[passIdForDrawWrapper] = new DrawWrapper(this._scene.getEngine());
            drawWrapper.setEffect(entry.mainDrawWrapper.effect, entry.mainDrawWrapper.defines);
        }
        return drawWrapper;
    }
    /**
     * Specifies that the submesh is ready to be used for depth rendering
     * @param subMesh submesh to check
     * @param defines the list of defines to take into account when checking the effect
     * @param shadowGenerator combined with subMesh, it defines the effect to check
     * @param useInstances specifies that instances should be used
     * @param passIdForDrawWrapper Id of the pass for which the draw wrapper should be created
     * @returns a boolean indicating that the submesh is ready or not
     */
    isReadyForSubMesh(subMesh, defines, shadowGenerator, useInstances, passIdForDrawWrapper) {
        if (this.standalone) {
            // will ensure the effect is (re)created for the base material
            if (!this._baseMaterial.isReadyForSubMesh(subMesh.getMesh(), subMesh, useInstances)) {
                return false;
            }
        }
        return this._makeEffect(subMesh, defines, shadowGenerator, passIdForDrawWrapper)?.isReady() ?? false;
    }
    /**
     * Disposes the resources
     */
    dispose() {
        this._baseMaterial.onEffectCreatedObservable.remove(this._onEffectCreatedObserver);
        this._onEffectCreatedObserver = null;
        const iterator = this._meshes.entries();
        for (let entry = iterator.next(); entry.done !== true; entry = iterator.next()) {
            const [mesh, observer] = entry.value;
            mesh.onDisposeObservable.remove(observer);
        }
    }
    _makeEffect(subMesh, defines, shadowGenerator, passIdForDrawWrapper) {
        const engine = this._scene.getEngine();
        const origEffectAndRenderPassId = this._subMeshToEffect.get(subMesh);
        if (!origEffectAndRenderPassId) {
            return null;
        }
        const [origEffect, origRenderPassId] = origEffectAndRenderPassId;
        if (!origEffect.isReady()) {
            return null;
        }
        let params = this._subMeshToDepthWrapper.get(subMesh, shadowGenerator);
        if (!params) {
            const mainDrawWrapper = new DrawWrapper(engine);
            mainDrawWrapper.defines = subMesh._getDrawWrapper(origRenderPassId)?.defines ?? null;
            params = {
                drawWrapper: [],
                mainDrawWrapper,
                depthDefines: "",
                token: RandomGUID(),
            };
            params.drawWrapper[passIdForDrawWrapper] = mainDrawWrapper;
            this._subMeshToDepthWrapper.set(subMesh, shadowGenerator, params);
        }
        const join = defines.join("\n");
        if (params.mainDrawWrapper.effect) {
            if (join === params.depthDefines) {
                // we already created the depth effect and it is still up to date for this submesh + shadow generator
                return params.mainDrawWrapper.effect;
            }
        }
        params.depthDefines = join;
        const uniforms = origEffect.getUniformNames().slice();
        // the depth effect is either out of date or has not been created yet
        let vertexCode = origEffect.vertexSourceCodeBeforeMigration, fragmentCode = origEffect.fragmentSourceCodeBeforeMigration;
        if (!vertexCode && !fragmentCode) {
            return null;
        }
        if (!this.doNotInjectCode) {
            // Declare the shadow map includes
            const vertexNormalBiasCode = this._options && this._options.remappedVariables
                ? `#include<shadowMapVertexNormalBias>(${this._options.remappedVariables.join(",")})`
                : `#include<shadowMapVertexNormalBias>`, vertexMetricCode = this._options && this._options.remappedVariables
                ? `#include<shadowMapVertexMetric>(${this._options.remappedVariables.join(",")})`
                : `#include<shadowMapVertexMetric>`, fragmentSoftTransparentShadow = this._options && this._options.remappedVariables
                ? `#include<shadowMapFragmentSoftTransparentShadow>(${this._options.remappedVariables.join(",")})`
                : `#include<shadowMapFragmentSoftTransparentShadow>`, fragmentBlockCode = `#include<shadowMapFragment>`, vertexExtraDeclartion = `#include<shadowMapVertexExtraDeclaration>`;
            // vertex code
            if (origEffect.shaderLanguage === 0 /* ShaderLanguage.GLSL */) {
                vertexCode = vertexCode.replace(/void\s+?main/g, `\n${vertexExtraDeclartion}\nvoid main`);
            }
            else {
                vertexCode = vertexCode.replace(/@vertex/g, `\n${vertexExtraDeclartion}\n@vertex`);
            }
            vertexCode = vertexCode.replace(/#define SHADOWDEPTH_NORMALBIAS|#define CUSTOM_VERTEX_UPDATE_WORLDPOS/g, vertexNormalBiasCode);
            if (vertexCode.indexOf("#define SHADOWDEPTH_METRIC") !== -1) {
                vertexCode = vertexCode.replace(/#define SHADOWDEPTH_METRIC/g, vertexMetricCode);
            }
            else {
                vertexCode = vertexCode.replace(/}\s*$/g, vertexMetricCode + "\n}");
            }
            vertexCode = vertexCode.replace(/#define SHADER_NAME.*?\n|out vec4 glFragColor;\n/g, "");
            // fragment code
            const hasLocationForSoftTransparentShadow = fragmentCode.indexOf("#define SHADOWDEPTH_SOFTTRANSPARENTSHADOW") >= 0 || fragmentCode.indexOf("#define CUSTOM_FRAGMENT_BEFORE_FOG") >= 0;
            const hasLocationForFragment = fragmentCode.indexOf("#define SHADOWDEPTH_FRAGMENT") !== -1;
            let fragmentCodeToInjectAtEnd = "";
            if (!hasLocationForSoftTransparentShadow) {
                fragmentCodeToInjectAtEnd = fragmentSoftTransparentShadow + "\n";
            }
            else {
                fragmentCode = fragmentCode.replace(/#define SHADOWDEPTH_SOFTTRANSPARENTSHADOW|#define CUSTOM_FRAGMENT_BEFORE_FOG/g, fragmentSoftTransparentShadow);
            }
            fragmentCode = fragmentCode.replace(/void\s+?main/g, Effect.IncludesShadersStore["shadowMapFragmentExtraDeclaration"] + "\nvoid main");
            if (hasLocationForFragment) {
                fragmentCode = fragmentCode.replace(/#define SHADOWDEPTH_FRAGMENT/g, fragmentBlockCode);
            }
            else {
                fragmentCodeToInjectAtEnd += fragmentBlockCode + "\n";
            }
            if (fragmentCodeToInjectAtEnd) {
                fragmentCode = fragmentCode.replace(/}\s*$/g, fragmentCodeToInjectAtEnd + "}");
            }
            uniforms.push("biasAndScaleSM", "depthValuesSM", "lightDataSM", "softTransparentShadowSM");
        }
        params.mainDrawWrapper.effect = engine.createEffect({
            vertexSource: vertexCode,
            fragmentSource: fragmentCode,
            vertexToken: params.token,
            fragmentToken: params.token,
        }, {
            attributes: origEffect.getAttributesNames(),
            uniformsNames: uniforms,
            uniformBuffersNames: origEffect.getUniformBuffersNames(),
            samplers: origEffect.getSamplers(),
            defines: join + "\n" + origEffect.defines.replace("#define SHADOWS", "").replace(/#define SHADOW\d/g, ""),
            indexParameters: origEffect.getIndexParameters(),
            shaderLanguage: origEffect.shaderLanguage,
        }, engine);
        for (let id = 0; id < params.drawWrapper.length; ++id) {
            if (id !== passIdForDrawWrapper) {
                params.drawWrapper[id]?.setEffect(params.mainDrawWrapper.effect, params.mainDrawWrapper.defines);
            }
        }
        return params.mainDrawWrapper.effect;
    }
}

// Do not edit.
const name$d = "gaussianSplattingFragmentDeclaration";
const shader$d = `vec4 gaussianColor(vec4 inColor)
{float A=-dot(vPosition,vPosition);if (A<-4.0) discard;float B=exp(A)*inColor.a;
#include<logDepthFragment>
vec3 color=inColor.rgb;
#ifdef FOG
#include<fogFragment>
#endif
return vec4(color,B);}
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name$d]) {
    ShaderStore.IncludesShadersStore[name$d] = shader$d;
}

// Do not edit.
const name$c = "gaussianSplattingPixelShader";
const shader$c = `#include<clipPlaneFragmentDeclaration>
#include<logDepthDeclaration>
#include<fogFragmentDeclaration>
varying vec4 vColor;varying vec2 vPosition;
#include<gaussianSplattingFragmentDeclaration>
void main () { 
#include<clipPlaneFragment>
gl_FragColor=gaussianColor(vColor);}
`;
// Sideeffect
if (!ShaderStore.ShadersStore[name$c]) {
    ShaderStore.ShadersStore[name$c] = shader$c;
}
/** @internal */
const gaussianSplattingPixelShader = { name: name$c, shader: shader$c };

const gaussianSplatting_fragment$1 = /*#__PURE__*/Object.freeze(/*#__PURE__*/Object.defineProperty({
    __proto__: null,
    gaussianSplattingPixelShader
}, Symbol.toStringTag, { value: 'Module' }));

// Do not edit.
const name$b = "gaussianSplattingVertexDeclaration";
const shader$b = `attribute vec3 position;attribute vec4 splatIndex0;attribute vec4 splatIndex1;attribute vec4 splatIndex2;attribute vec4 splatIndex3;uniform mat4 view;uniform mat4 projection;uniform mat4 world;uniform vec4 vEyePosition;`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name$b]) {
    ShaderStore.IncludesShadersStore[name$b] = shader$b;
}

// Do not edit.
const name$a = "gaussianSplattingUboDeclaration";
const shader$a = `#include<sceneUboDeclaration>
#include<meshUboDeclaration>
attribute vec3 position;attribute vec4 splatIndex0;attribute vec4 splatIndex1;attribute vec4 splatIndex2;attribute vec4 splatIndex3;
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name$a]) {
    ShaderStore.IncludesShadersStore[name$a] = shader$a;
}

// Do not edit.
const name$9 = "gaussianSplatting";
const shader$9 = `#if !defined(WEBGL2) && !defined(WEBGPU) && !defined(NATIVE)
mat3 transpose(mat3 matrix) {return mat3(matrix[0][0],matrix[1][0],matrix[2][0],
matrix[0][1],matrix[1][1],matrix[2][1],
matrix[0][2],matrix[1][2],matrix[2][2]);}
#endif
vec2 getDataUV(float index,vec2 textureSize) {float y=floor(index/textureSize.x);float x=index-y*textureSize.x;return vec2((x+0.5)/textureSize.x,(y+0.5)/textureSize.y);}
#if SH_DEGREE>0
ivec2 getDataUVint(float index,vec2 textureSize) {float y=floor(index/textureSize.x);float x=index-y*textureSize.x;return ivec2(uint(x+0.5),uint(y+0.5));}
#endif
struct Splat {vec4 center;vec4 color;vec4 covA;vec4 covB;
#if SH_DEGREE>0
uvec4 sh0; 
#endif
#if SH_DEGREE>1
uvec4 sh1;
#endif
#if SH_DEGREE>2
uvec4 sh2;
#endif
};float getSplatIndex(int localIndex)
{float splatIndex;switch (localIndex)
{case 0: splatIndex=splatIndex0.x; break;case 1: splatIndex=splatIndex0.y; break;case 2: splatIndex=splatIndex0.z; break;case 3: splatIndex=splatIndex0.w; break;case 4: splatIndex=splatIndex1.x; break;case 5: splatIndex=splatIndex1.y; break;case 6: splatIndex=splatIndex1.z; break;case 7: splatIndex=splatIndex1.w; break;case 8: splatIndex=splatIndex2.x; break;case 9: splatIndex=splatIndex2.y; break;case 10: splatIndex=splatIndex2.z; break;case 11: splatIndex=splatIndex2.w; break;case 12: splatIndex=splatIndex3.x; break;case 13: splatIndex=splatIndex3.y; break;case 14: splatIndex=splatIndex3.z; break;case 15: splatIndex=splatIndex3.w; break;}
return splatIndex;}
Splat readSplat(float splatIndex)
{Splat splat;vec2 splatUV=getDataUV(splatIndex,dataTextureSize);splat.center=texture2D(centersTexture,splatUV);splat.color=texture2D(colorsTexture,splatUV);splat.covA=texture2D(covariancesATexture,splatUV)*splat.center.w;splat.covB=texture2D(covariancesBTexture,splatUV)*splat.center.w;
#if SH_DEGREE>0
ivec2 splatUVint=getDataUVint(splatIndex,dataTextureSize);splat.sh0=texelFetch(shTexture0,splatUVint,0);
#endif
#if SH_DEGREE>1
splat.sh1=texelFetch(shTexture1,splatUVint,0);
#endif
#if SH_DEGREE>2
splat.sh2=texelFetch(shTexture2,splatUVint,0);
#endif
return splat;}
#if defined(WEBGL2) || defined(WEBGPU) || defined(NATIVE)
vec3 computeColorFromSHDegree(vec3 dir,const vec3 sh[16])
{const float SH_C0=0.28209479;const float SH_C1=0.48860251;float SH_C2[5];SH_C2[0]=1.092548430;SH_C2[1]=-1.09254843;SH_C2[2]=0.315391565;SH_C2[3]=-1.09254843;SH_C2[4]=0.546274215;float SH_C3[7];SH_C3[0]=-0.59004358;SH_C3[1]=2.890611442;SH_C3[2]=-0.45704579;SH_C3[3]=0.373176332;SH_C3[4]=-0.45704579;SH_C3[5]=1.445305721;SH_C3[6]=-0.59004358;vec3 result=/*SH_C0**/sh[0];
#if SH_DEGREE>0
float x=dir.x;float y=dir.y;float z=dir.z;result+=- SH_C1*y*sh[1]+SH_C1*z*sh[2]-SH_C1*x*sh[3];
#if SH_DEGREE>1
float xx=x*x,yy=y*y,zz=z*z;float xy=x*y,yz=y*z,xz=x*z;result+=
SH_C2[0]*xy*sh[4] +
SH_C2[1]*yz*sh[5] +
SH_C2[2]*(2.0*zz-xx-yy)*sh[6] +
SH_C2[3]*xz*sh[7] +
SH_C2[4]*(xx-yy)*sh[8];
#if SH_DEGREE>2
result+=
SH_C3[0]*y*(3.0*xx-yy)*sh[9] +
SH_C3[1]*xy*z*sh[10] +
SH_C3[2]*y*(4.0*zz-xx-yy)*sh[11] +
SH_C3[3]*z*(2.0*zz-3.0*xx-3.0*yy)*sh[12] +
SH_C3[4]*x*(4.0*zz-xx-yy)*sh[13] +
SH_C3[5]*z*(xx-yy)*sh[14] +
SH_C3[6]*x*(xx-3.0*yy)*sh[15];
#endif
#endif
#endif
return result;}
vec4 decompose(uint value)
{vec4 components=vec4(
float((value ) & 255u),
float((value>>uint( 8)) & 255u),
float((value>>uint(16)) & 255u),
float((value>>uint(24)) & 255u));return components*vec4(2./255.)-vec4(1.);}
vec3 computeSH(Splat splat,vec3 dir)
{vec3 sh[16];sh[0]=vec3(0.,0.,0.);
#if SH_DEGREE>0
vec4 sh00=decompose(splat.sh0.x);vec4 sh01=decompose(splat.sh0.y);vec4 sh02=decompose(splat.sh0.z);sh[1]=vec3(sh00.x,sh00.y,sh00.z);sh[2]=vec3(sh00.w,sh01.x,sh01.y);sh[3]=vec3(sh01.z,sh01.w,sh02.x);
#endif
#if SH_DEGREE>1
vec4 sh03=decompose(splat.sh0.w);vec4 sh04=decompose(splat.sh1.x);vec4 sh05=decompose(splat.sh1.y);sh[4]=vec3(sh02.y,sh02.z,sh02.w);sh[5]=vec3(sh03.x,sh03.y,sh03.z);sh[6]=vec3(sh03.w,sh04.x,sh04.y);sh[7]=vec3(sh04.z,sh04.w,sh05.x);sh[8]=vec3(sh05.y,sh05.z,sh05.w);
#endif
#if SH_DEGREE>2
vec4 sh06=decompose(splat.sh1.z);vec4 sh07=decompose(splat.sh1.w);vec4 sh08=decompose(splat.sh2.x);vec4 sh09=decompose(splat.sh2.y);vec4 sh10=decompose(splat.sh2.z);vec4 sh11=decompose(splat.sh2.w);sh[9]=vec3(sh06.x,sh06.y,sh06.z);sh[10]=vec3(sh06.w,sh07.x,sh07.y);sh[11]=vec3(sh07.z,sh07.w,sh08.x);sh[12]=vec3(sh08.y,sh08.z,sh08.w);sh[13]=vec3(sh09.x,sh09.y,sh09.z);sh[14]=vec3(sh09.w,sh10.x,sh10.y);sh[15]=vec3(sh10.z,sh10.w,sh11.x); 
#endif
return computeColorFromSHDegree(dir,sh);}
#else
vec3 computeSH(Splat splat,vec3 dir)
{return vec3(0.,0.,0.);}
#endif
vec4 gaussianSplatting(vec2 meshPos,vec3 worldPos,vec2 scale,vec3 covA,vec3 covB,mat4 worldMatrix,mat4 viewMatrix,mat4 projectionMatrix)
{mat4 modelView=viewMatrix*worldMatrix;vec4 camspace=viewMatrix*vec4(worldPos,1.);vec4 pos2d=projectionMatrix*camspace;float bounds=1.2*pos2d.w;if (pos2d.z<-pos2d.w || pos2d.x<-bounds || pos2d.x>bounds
|| pos2d.y<-bounds || pos2d.y>bounds) {return vec4(0.0,0.0,2.0,1.0);}
mat3 Vrk=mat3(
covA.x,covA.y,covA.z,
covA.y,covB.x,covB.y,
covA.z,covB.y,covB.z
);bool isOrtho=abs(projectionMatrix[3][3]-1.0)<0.001;mat3 J;if (isOrtho) {J=mat3(
focal.x,0.,0.,
0.,focal.y,0.,
0.,0.,0.
);} else {J=mat3(
focal.x/camspace.z,0.,-(focal.x*camspace.x)/(camspace.z*camspace.z),
0.,focal.y/camspace.z,-(focal.y*camspace.y)/(camspace.z*camspace.z),
0.,0.,0.
);}
mat3 T=transpose(mat3(modelView))*J;mat3 cov2d=transpose(T)*Vrk*T;
#if COMPENSATION
float c00=cov2d[0][0];float c11=cov2d[1][1];float c01=cov2d[0][1];float detOrig=c00*c11-c01*c01;
#endif
cov2d[0][0]+=kernelSize;cov2d[1][1]+=kernelSize;
#if COMPENSATION
vec3 c2d=vec3(cov2d[0][0],c01,cov2d[1][1]);float detBlur=c2d.x*c2d.z-c2d.y*c2d.y;float compensation=sqrt(max(0.,detOrig/detBlur));vColor.w*=compensation;
#endif
float mid=(cov2d[0][0]+cov2d[1][1])/2.0;float radius=length(vec2((cov2d[0][0]-cov2d[1][1])/2.0,cov2d[0][1]));float epsilon=0.0001;float lambda1=mid+radius+epsilon,lambda2=mid-radius+epsilon;if (lambda2<0.0)
{return vec4(0.0,0.0,2.0,1.0);}
vec2 diagonalVector=normalize(vec2(cov2d[0][1],lambda1-cov2d[0][0]));vec2 majorAxis=min(sqrt(2.0*lambda1),1024.0)*diagonalVector;vec2 minorAxis=min(sqrt(2.0*lambda2),1024.0)*vec2(diagonalVector.y,-diagonalVector.x);vec2 vCenter=vec2(pos2d);float scaleFactor=isOrtho ? 1.0 : pos2d.w;return vec4(
vCenter 
+ ((meshPos.x*majorAxis
+ meshPos.y*minorAxis)*invViewport*scaleFactor)*scale,pos2d.zw);}`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name$9]) {
    ShaderStore.IncludesShadersStore[name$9] = shader$9;
}

// Do not edit.
const name$8 = "gaussianSplattingVertexShader";
const shader$8 = `#include<__decl__gaussianSplattingVertex>
#ifdef LOGARITHMICDEPTH
#extension GL_EXT_frag_depth : enable
#endif
#include<clipPlaneVertexDeclaration>
#include<fogVertexDeclaration>
#include<logDepthDeclaration>
#include<helperFunctions>
uniform vec2 invViewport;uniform vec2 dataTextureSize;uniform vec2 focal;uniform float kernelSize;uniform vec3 eyePosition;uniform float alpha;uniform sampler2D covariancesATexture;uniform sampler2D covariancesBTexture;uniform sampler2D centersTexture;uniform sampler2D colorsTexture;
#if SH_DEGREE>0
uniform highp usampler2D shTexture0;
#endif
#if SH_DEGREE>1
uniform highp usampler2D shTexture1;
#endif
#if SH_DEGREE>2
uniform highp usampler2D shTexture2;
#endif
varying vec4 vColor;varying vec2 vPosition;
#include<gaussianSplatting>
void main () {float splatIndex=getSplatIndex(int(position.z+0.5));Splat splat=readSplat(splatIndex);vec3 covA=splat.covA.xyz;vec3 covB=vec3(splat.covA.w,splat.covB.xy);vec4 worldPos=world*vec4(splat.center.xyz,1.0);vColor=splat.color;vPosition=position.xy;
#if SH_DEGREE>0
mat3 worldRot=mat3(world);mat3 normWorldRot=inverseMat3(worldRot);vec3 eyeToSplatLocalSpace=normalize(normWorldRot*(worldPos.xyz-eyePosition));vColor.xyz=splat.color.xyz+computeSH(splat,eyeToSplatLocalSpace);
#endif
vColor.w*=alpha;gl_Position=gaussianSplatting(position.xy,worldPos.xyz,vec2(1.,1.),covA,covB,world,view,projection);
#include<clipPlaneVertex>
#include<fogVertex>
#include<logDepthVertex>
}
`;
// Sideeffect
if (!ShaderStore.ShadersStore[name$8]) {
    ShaderStore.ShadersStore[name$8] = shader$8;
}
/** @internal */
const gaussianSplattingVertexShader = { name: name$8, shader: shader$8 };

const gaussianSplatting_vertex$1 = /*#__PURE__*/Object.freeze(/*#__PURE__*/Object.defineProperty({
    __proto__: null,
    gaussianSplattingVertexShader
}, Symbol.toStringTag, { value: 'Module' }));

// Do not edit.
const name$7 = "gaussianSplattingFragmentDeclaration";
const shader$7 = `fn gaussianColor(inColor: vec4f,inPosition: vec2f)->vec4f
{var A : f32=-dot(inPosition,inPosition);if (A>-4.0)
{var B: f32=exp(A)*inColor.a;
#include<logDepthFragment>
var color: vec3f=inColor.rgb;
#ifdef FOG
#include<fogFragment>
#endif
return vec4f(color,B);} else {return vec4f(0.0);}}
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStoreWGSL[name$7]) {
    ShaderStore.IncludesShadersStoreWGSL[name$7] = shader$7;
}

// Do not edit.
const name$6 = "gaussianSplattingPixelShader";
const shader$6 = `#include<clipPlaneFragmentDeclaration>
#include<logDepthDeclaration>
#include<fogFragmentDeclaration>
varying vColor: vec4f;varying vPosition: vec2f;
#include<gaussianSplattingFragmentDeclaration>
@fragment
fn main(input: FragmentInputs)->FragmentOutputs {
#include<clipPlaneFragment>
fragmentOutputs.color=gaussianColor(input.vColor,input.vPosition);}
`;
// Sideeffect
if (!ShaderStore.ShadersStoreWGSL[name$6]) {
    ShaderStore.ShadersStoreWGSL[name$6] = shader$6;
}
/** @internal */
const gaussianSplattingPixelShaderWGSL = { name: name$6, shader: shader$6 };

const gaussianSplatting_fragment = /*#__PURE__*/Object.freeze(/*#__PURE__*/Object.defineProperty({
    __proto__: null,
    gaussianSplattingPixelShaderWGSL
}, Symbol.toStringTag, { value: 'Module' }));

// Do not edit.
const name$5 = "gaussianSplatting";
const shader$5 = `fn getDataUV(index: f32,dataTextureSize: vec2f)->vec2<f32> {let y: f32=floor(index/dataTextureSize.x);let x: f32=index-y*dataTextureSize.x;return vec2f((x+0.5),(y+0.5));}
struct Splat {center: vec4f,
color: vec4f,
covA: vec4f,
covB: vec4f,
#if SH_DEGREE>0
sh0: vec4<u32>,
#endif
#if SH_DEGREE>1
sh1: vec4<u32>,
#endif
#if SH_DEGREE>2
sh2: vec4<u32>,
#endif
};fn getSplatIndex(localIndex: i32,splatIndex0: vec4f,splatIndex1: vec4f,splatIndex2: vec4f,splatIndex3: vec4f)->f32 {var splatIndex: f32;switch (localIndex)
{case 0:
{splatIndex=splatIndex0.x;break;}
case 1:
{splatIndex=splatIndex0.y;break;}
case 2:
{splatIndex=splatIndex0.z;break;}
case 3:
{splatIndex=splatIndex0.w;break;}
case 4:
{splatIndex=splatIndex1.x;break;}
case 5:
{splatIndex=splatIndex1.y;break;}
case 6:
{splatIndex=splatIndex1.z;break;}
case 7:
{splatIndex=splatIndex1.w;break;}
case 8:
{splatIndex=splatIndex2.x;break;}
case 9:
{splatIndex=splatIndex2.y;break;}
case 10:
{splatIndex=splatIndex2.z;break;}
case 11:
{splatIndex=splatIndex2.w;break;}
case 12:
{splatIndex=splatIndex3.x;break;}
case 13:
{splatIndex=splatIndex3.y;break;}
case 14:
{splatIndex=splatIndex3.z;break;}
default:
{splatIndex=splatIndex3.w;break;}}
return splatIndex;}
fn readSplat(splatIndex: f32,dataTextureSize: vec2f)->Splat {var splat: Splat;let splatUV=getDataUV(splatIndex,dataTextureSize);let splatUVi32=vec2<i32>(i32(splatUV.x),i32(splatUV.y));splat.center=textureLoad(centersTexture,splatUVi32,0);splat.color=textureLoad(colorsTexture,splatUVi32,0);splat.covA=textureLoad(covariancesATexture,splatUVi32,0)*splat.center.w;splat.covB=textureLoad(covariancesBTexture,splatUVi32,0)*splat.center.w;
#if SH_DEGREE>0
splat.sh0=textureLoad(shTexture0,splatUVi32,0);
#endif
#if SH_DEGREE>1
splat.sh1=textureLoad(shTexture1,splatUVi32,0);
#endif
#if SH_DEGREE>2
splat.sh2=textureLoad(shTexture2,splatUVi32,0);
#endif
return splat;}
fn computeColorFromSHDegree(dir: vec3f,sh: array<vec3<f32>,16>)->vec3f
{let SH_C0: f32=0.28209479;let SH_C1: f32=0.48860251;var SH_C2: array<f32,5>=array<f32,5>(
1.092548430,
-1.09254843,
0.315391565,
-1.09254843,
0.546274215
);var SH_C3: array<f32,7>=array<f32,7>(
-0.59004358,
2.890611442,
-0.45704579,
0.373176332,
-0.45704579,
1.445305721,
-0.59004358
);var result: vec3f=/*SH_C0**/sh[0];
#if SH_DEGREE>0
let x: f32=dir.x;let y: f32=dir.y;let z: f32=dir.z;result+=-SH_C1*y*sh[1]+SH_C1*z*sh[2]-SH_C1*x*sh[3];
#if SH_DEGREE>1
let xx: f32=x*x;let yy: f32=y*y;let zz: f32=z*z;let xy: f32=x*y;let yz: f32=y*z;let xz: f32=x*z;result+=
SH_C2[0]*xy*sh[4] +
SH_C2[1]*yz*sh[5] +
SH_C2[2]*(2.0f*zz-xx-yy)*sh[6] +
SH_C2[3]*xz*sh[7] +
SH_C2[4]*(xx-yy)*sh[8];
#if SH_DEGREE>2
result+=
SH_C3[0]*y*(3.0f*xx-yy)*sh[9] +
SH_C3[1]*xy*z*sh[10] +
SH_C3[2]*y*(4.0f*zz-xx-yy)*sh[11] +
SH_C3[3]*z*(2.0f*zz-3.0f*xx-3.0f*yy)*sh[12] +
SH_C3[4]*x*(4.0f*zz-xx-yy)*sh[13] +
SH_C3[5]*z*(xx-yy)*sh[14] +
SH_C3[6]*x*(xx-3.0f*yy)*sh[15];
#endif
#endif
#endif
return result;}
fn decompose(value: u32)->vec4f
{let components : vec4f=vec4f(
f32((value ) & 255u),
f32((value>>u32( 8)) & 255u),
f32((value>>u32(16)) & 255u),
f32((value>>u32(24)) & 255u));return components*vec4f(2./255.)-vec4f(1.);}
fn computeSH(splat: Splat,dir: vec3f)->vec3f
{var sh: array<vec3<f32>,16>;sh[0]=vec3f(0.,0.,0.);
#if SH_DEGREE>0
let sh00: vec4f=decompose(splat.sh0.x);let sh01: vec4f=decompose(splat.sh0.y);let sh02: vec4f=decompose(splat.sh0.z);sh[1]=vec3f(sh00.x,sh00.y,sh00.z);sh[2]=vec3f(sh00.w,sh01.x,sh01.y);sh[3]=vec3f(sh01.z,sh01.w,sh02.x);
#endif
#if SH_DEGREE>1
let sh03: vec4f=decompose(splat.sh0.w);let sh04: vec4f=decompose(splat.sh1.x);let sh05: vec4f=decompose(splat.sh1.y);sh[4]=vec3f(sh02.y,sh02.z,sh02.w);sh[5]=vec3f(sh03.x,sh03.y,sh03.z);sh[6]=vec3f(sh03.w,sh04.x,sh04.y);sh[7]=vec3f(sh04.z,sh04.w,sh05.x);sh[8]=vec3f(sh05.y,sh05.z,sh05.w);
#endif
#if SH_DEGREE>2
let sh06: vec4f=decompose(splat.sh1.z);let sh07: vec4f=decompose(splat.sh1.w);let sh08: vec4f=decompose(splat.sh2.x);let sh09: vec4f=decompose(splat.sh2.y);let sh10: vec4f=decompose(splat.sh2.z);let sh11: vec4f=decompose(splat.sh2.w);sh[9]=vec3f(sh06.x,sh06.y,sh06.z);sh[10]=vec3f(sh06.w,sh07.x,sh07.y);sh[11]=vec3f(sh07.z,sh07.w,sh08.x);sh[12]=vec3f(sh08.y,sh08.z,sh08.w);sh[13]=vec3f(sh09.x,sh09.y,sh09.z);sh[14]=vec3f(sh09.w,sh10.x,sh10.y);sh[15]=vec3f(sh10.z,sh10.w,sh11.x); 
#endif
return computeColorFromSHDegree(dir,sh);}
fn gaussianSplatting(
meshPos: vec2<f32>,
worldPos: vec3<f32>,
scale: vec2<f32>,
covA: vec3<f32>,
covB: vec3<f32>,
worldMatrix: mat4x4<f32>,
viewMatrix: mat4x4<f32>,
projectionMatrix: mat4x4<f32>,
focal: vec2f,
invViewport: vec2f,
kernelSize: f32
)->vec4f {let modelView=viewMatrix*worldMatrix;let camspace=viewMatrix*vec4f(worldPos,1.0);let pos2d=projectionMatrix*camspace;let bounds=1.2*pos2d.w;if (pos2d.z<0. || pos2d.x<-bounds || pos2d.x>bounds || pos2d.y<-bounds || pos2d.y>bounds) {return vec4f(0.0,0.0,2.0,1.0);}
let Vrk=mat3x3<f32>(
covA.x,covA.y,covA.z,
covA.y,covB.x,covB.y,
covA.z,covB.y,covB.z
);let isOrtho=abs(projectionMatrix[3][3]-1.0)<0.001;var J: mat3x3<f32>;if (isOrtho) {J=mat3x3<f32>(
focal.x,0.0,0.0,
0.0,focal.y,0.0,
0.0,0.0,0.0
);} else {J=mat3x3<f32>(
focal.x/camspace.z,0.0,-(focal.x*camspace.x)/(camspace.z*camspace.z),
0.0,focal.y/camspace.z,-(focal.y*camspace.y)/(camspace.z*camspace.z),
0.0,0.0,0.0
);}
let T=transpose(mat3x3<f32>(
modelView[0].xyz,
modelView[1].xyz,
modelView[2].xyz))*J;var cov2d=transpose(T)*Vrk*T;
#if COMPENSATION
let c00: f32=cov2d[0][0];let c11: f32=cov2d[1][1];let c01: f32=cov2d[0][1];let detOrig: f32=c00*c11-c01*c01;
#endif
cov2d[0][0]+=kernelSize;cov2d[1][1]+=kernelSize;
#if COMPENSATION
let c2d: vec3f=vec3f(cov2d[0][0],c01,cov2d[1][1]);let detBlur: f32=c2d.x*c2d.z-c2d.y*c2d.y;let compensation: f32=sqrt(max(0.,detOrig/detBlur));vertexOutputs.vColor.w*=compensation;
#endif
let mid=(cov2d[0][0]+cov2d[1][1])/2.0;let radius=length(vec2<f32>((cov2d[0][0]-cov2d[1][1])/2.0,cov2d[0][1]));let lambda1=mid+radius;let lambda2=mid-radius;if (lambda2<0.0) {return vec4f(0.0,0.0,2.0,1.0);}
let diagonalVector=normalize(vec2<f32>(cov2d[0][1],lambda1-cov2d[0][0]));let majorAxis=min(sqrt(2.0*lambda1),1024.0)*diagonalVector;let minorAxis=min(sqrt(2.0*lambda2),1024.0)*vec2<f32>(diagonalVector.y,-diagonalVector.x);let vCenter=vec2<f32>(pos2d.x,pos2d.y);let scaleFactor=select(pos2d.w,1.0,isOrtho);return vec4f(
vCenter+((meshPos.x*majorAxis+meshPos.y*minorAxis)*invViewport*scaleFactor)*scale,
pos2d.z,
pos2d.w
);}`;
// Sideeffect
if (!ShaderStore.IncludesShadersStoreWGSL[name$5]) {
    ShaderStore.IncludesShadersStoreWGSL[name$5] = shader$5;
}

// Do not edit.
const name$4 = "gaussianSplattingVertexShader";
const shader$4 = `#include<sceneUboDeclaration>
#include<meshUboDeclaration>
#include<helperFunctions>
#include<clipPlaneVertexDeclaration>
#include<fogVertexDeclaration>
#include<logDepthDeclaration>
attribute splatIndex0: vec4f;attribute splatIndex1: vec4f;attribute splatIndex2: vec4f;attribute splatIndex3: vec4f;attribute position: vec3f;uniform invViewport: vec2f;uniform dataTextureSize: vec2f;uniform focal: vec2f;uniform kernelSize: f32;uniform eyePosition: vec3f;uniform alpha: f32;var covariancesATexture: texture_2d<f32>;var covariancesBTexture: texture_2d<f32>;var centersTexture: texture_2d<f32>;var colorsTexture: texture_2d<f32>;
#if SH_DEGREE>0
var shTexture0: texture_2d<u32>;
#endif
#if SH_DEGREE>1
var shTexture1: texture_2d<u32>;
#endif
#if SH_DEGREE>2
var shTexture2: texture_2d<u32>;
#endif
varying vColor: vec4f;varying vPosition: vec2f;
#include<gaussianSplatting>
@vertex
fn main(input : VertexInputs)->FragmentInputs {let splatIndex: f32=getSplatIndex(i32(input.position.z+0.5),input.splatIndex0,input.splatIndex1,input.splatIndex2,input.splatIndex3);var splat: Splat=readSplat(splatIndex,uniforms.dataTextureSize);var covA: vec3f=splat.covA.xyz;var covB: vec3f=vec3f(splat.covA.w,splat.covB.xy);let worldPos: vec4f=mesh.world*vec4f(splat.center.xyz,1.0);vertexOutputs.vPosition=input.position.xy;
#if SH_DEGREE>0
let worldRot: mat3x3f= mat3x3f(mesh.world[0].xyz,mesh.world[1].xyz,mesh.world[2].xyz);let normWorldRot: mat3x3f=inverseMat3(worldRot);var eyeToSplatLocalSpace: vec3f=normalize(normWorldRot*(worldPos.xyz-uniforms.eyePosition.xyz));vertexOutputs.vColor=vec4f(splat.color.xyz+computeSH(splat,eyeToSplatLocalSpace),splat.color.w*uniforms.alpha);
#else
vertexOutputs.vColor=vec4f(splat.color.xyz,splat.color.w*uniforms.alpha);
#endif
vertexOutputs.position=gaussianSplatting(input.position.xy,worldPos.xyz,vec2f(1.0,1.0),covA,covB,mesh.world,scene.view,scene.projection,uniforms.focal,uniforms.invViewport,uniforms.kernelSize);
#include<clipPlaneVertex>
#include<fogVertex>
#include<logDepthVertex>
}
`;
// Sideeffect
if (!ShaderStore.ShadersStoreWGSL[name$4]) {
    ShaderStore.ShadersStoreWGSL[name$4] = shader$4;
}
/** @internal */
const gaussianSplattingVertexShaderWGSL = { name: name$4, shader: shader$4 };

const gaussianSplatting_vertex = /*#__PURE__*/Object.freeze(/*#__PURE__*/Object.defineProperty({
    __proto__: null,
    gaussianSplattingVertexShaderWGSL
}, Symbol.toStringTag, { value: 'Module' }));

// Do not edit.
const name$3 = "gaussianSplattingDepthPixelShader";
const shader$3 = `precision highp float;varying vec2 vPosition;varying vec4 vColor;
#ifdef DEPTH_RENDER
varying float vDepthMetric;
#endif
void main(void) {float A=-dot(vPosition,vPosition);
#if defined(SM_SOFTTRANSPARENTSHADOW) && SM_SOFTTRANSPARENTSHADOW==1
float alpha=exp(A)*vColor.a;if (A<-4.) discard;
#else
if (A<-vColor.a) discard;
#endif
#ifdef DEPTH_RENDER
gl_FragColor=vec4(vDepthMetric,0.0,0.0,1.0);
#endif
}`;
// Sideeffect
if (!ShaderStore.ShadersStore[name$3]) {
    ShaderStore.ShadersStore[name$3] = shader$3;
}

// Do not edit.
const name$2 = "gaussianSplattingDepthVertexShader";
const shader$2 = `#include<__decl__gaussianSplattingVertex>
uniform vec2 invViewport;uniform vec2 dataTextureSize;uniform vec2 focal;uniform float kernelSize;uniform float alpha;uniform sampler2D covariancesATexture;uniform sampler2D covariancesBTexture;uniform sampler2D centersTexture;uniform sampler2D colorsTexture;varying vec2 vPosition;varying vec4 vColor;
#include<gaussianSplatting>
#ifdef DEPTH_RENDER
uniform vec2 depthValues;varying float vDepthMetric;
#endif
void main(void) {float splatIndex=getSplatIndex(int(position.z+0.5));Splat splat=readSplat(splatIndex);vec3 covA=splat.covA.xyz;vec3 covB=vec3(splat.covA.w,splat.covB.xy);vec4 worldPosGS=world*vec4(splat.center.xyz,1.0);vPosition=position.xy;vColor=splat.color;vColor.w*=alpha;gl_Position=gaussianSplatting(position.xy,worldPosGS.xyz,vec2(1.,1.),covA,covB,world,view,projection);
#ifdef DEPTH_RENDER
#ifdef USE_REVERSE_DEPTHBUFFER
vDepthMetric=((-gl_Position.z+depthValues.x)/(depthValues.y));
#else
vDepthMetric=((gl_Position.z+depthValues.x)/(depthValues.y));
#endif
#endif
}`;
// Sideeffect
if (!ShaderStore.ShadersStore[name$2]) {
    ShaderStore.ShadersStore[name$2] = shader$2;
}

// Do not edit.
const name$1 = "gaussianSplattingDepthPixelShader";
const shader$1 = `#include<gaussianSplattingFragmentDeclaration>
varying vPosition: vec2f;varying vColor: vec4f;
#ifdef DEPTH_RENDER
varying vDepthMetric: f32;
#endif
fn checkDiscard(inPosition: vec2f,inColor: vec4f)->vec4f {var A : f32=-dot(inPosition,inPosition);var alpha : f32=exp(A)*inColor.a;
#if defined(SM_SOFTTRANSPARENTSHADOW) && SM_SOFTTRANSPARENTSHADOW==1
if (A<-4.) {discard;}
#else
if (A<-inColor.a) {discard;}
#endif
#ifdef DEPTH_RENDER
return vec4f(fragmentInputs.vDepthMetric,0.0,0.0,1.0);
#else
return vec4f(inColor.rgb,alpha);
#endif
}
#define CUSTOM_FRAGMENT_DEFINITIONS
@fragment
fn main(input: FragmentInputs)->FragmentOutputs {fragmentOutputs.color=checkDiscard(fragmentInputs.vPosition,fragmentInputs.vColor);
#if defined(SM_SOFTTRANSPARENTSHADOW) && SM_SOFTTRANSPARENTSHADOW==1
var alpha : f32=fragmentOutputs.color.a;
#endif
}
`;
// Sideeffect
if (!ShaderStore.ShadersStoreWGSL[name$1]) {
    ShaderStore.ShadersStoreWGSL[name$1] = shader$1;
}

// Do not edit.
const name = "gaussianSplattingDepthVertexShader";
const shader = `#include<sceneUboDeclaration>
#include<meshUboDeclaration>
attribute splatIndex0: vec4f;attribute splatIndex1: vec4f;attribute splatIndex2: vec4f;attribute splatIndex3: vec4f;attribute position: vec3f;uniform invViewport: vec2f;uniform dataTextureSize: vec2f;uniform focal: vec2f;uniform kernelSize: f32;uniform alpha: f32;var covariancesATexture: texture_2d<f32>;var covariancesBTexture: texture_2d<f32>;var centersTexture: texture_2d<f32>;var colorsTexture: texture_2d<f32>;varying vPosition: vec2f;varying vColor: vec4f;
#ifdef DEPTH_RENDER
uniform depthValues: vec2f;varying vDepthMetric: f32;
#endif
#include<gaussianSplatting>
@vertex
fn main(input : VertexInputs)->FragmentInputs {let splatIndex: f32=getSplatIndex(i32(input.position.z+0.5),input.splatIndex0,input.splatIndex1,input.splatIndex2,input.splatIndex3);var splat: Splat=readSplat(splatIndex,uniforms.dataTextureSize);var covA: vec3f=splat.covA.xyz;var covB: vec3f=vec3f(splat.covA.w,splat.covB.xy);let worldPos: vec4f=mesh.world*vec4f(splat.center.xyz,1.0);vertexOutputs.vPosition=input.position.xy;vertexOutputs.vColor=splat.color;vertexOutputs.vColor.w*=uniforms.alpha;vertexOutputs.position=gaussianSplatting(input.position.xy,worldPos.xyz,vec2f(1.0,1.0),covA,covB,mesh.world,scene.view,scene.projection,uniforms.focal,uniforms.invViewport,uniforms.kernelSize);
#ifdef DEPTH_RENDER
#ifdef USE_REVERSE_DEPTHBUFFER
vertexOutputs.vDepthMetric=((-vertexOutputs.position.z+uniforms.depthValues.x)/(uniforms.depthValues.y));
#else
vertexOutputs.vDepthMetric=((vertexOutputs.position.z+uniforms.depthValues.x)/(uniforms.depthValues.y));
#endif
#endif
}`;
// Sideeffect
if (!ShaderStore.ShadersStoreWGSL[name]) {
    ShaderStore.ShadersStoreWGSL[name] = shader;
}

/**
 * @internal
 */
class GaussianSplattingMaterialDefines extends MaterialDefines {
    /**
     * Constructor of the defines.
     */
    constructor() {
        super();
        this.FOG = false;
        this.THIN_INSTANCES = true;
        this.LOGARITHMICDEPTH = false;
        this.CLIPPLANE = false;
        this.CLIPPLANE2 = false;
        this.CLIPPLANE3 = false;
        this.CLIPPLANE4 = false;
        this.CLIPPLANE5 = false;
        this.CLIPPLANE6 = false;
        this.SH_DEGREE = 0;
        this.COMPENSATION = false;
        this.rebuild();
    }
}
/**
 * GaussianSplattingMaterial material used to render Gaussian Splatting
 * @experimental
 */
class GaussianSplattingMaterial extends PushMaterial {
    /**
     * Instantiates a Gaussian Splatting Material in the given scene
     * @param name The friendly name of the material
     * @param scene The scene to add the material to
     */
    constructor(name, scene) {
        super(name, scene);
        /**
         * Point spread function (default 0.3). Can be overriden per GS material, otherwise, using default static `KernelSize` value
         */
        this.kernelSize = GaussianSplattingMaterial.KernelSize;
        this._compensation = GaussianSplattingMaterial.Compensation;
        // set to true when material defines are dirty
        this._isDirty = false;
        this._sourceMesh = null;
        this.backFaceCulling = false;
        this.shadowDepthWrapper = GaussianSplattingMaterial._MakeGaussianSplattingShadowDepthWrapper(scene, this.shaderLanguage);
    }
    /**
     * Set compensation default value is `GaussianSplattingMaterial.Compensation`
     */
    set compensation(value) {
        this._isDirty = this._isDirty != value;
        this._compensation = value;
    }
    /**
     * Get compensation
     */
    get compensation() {
        return this._compensation;
    }
    /**
     * Gets a boolean indicating that current material needs to register RTT
     */
    get hasRenderTargetTextures() {
        return false;
    }
    /**
     * Specifies whether or not this material should be rendered in alpha test mode.
     * @returns false
     */
    needAlphaTesting() {
        return false;
    }
    /**
     * Specifies whether or not this material should be rendered in alpha blend mode.
     * @returns true
     */
    needAlphaBlending() {
        return true;
    }
    /**
     * Checks whether the material is ready to be rendered for a given mesh.
     * @param mesh The mesh to render
     * @param subMesh The submesh to check against
     * @returns true if all the dependencies are ready (Textures, Effects...)
     */
    isReadyForSubMesh(mesh, subMesh) {
        const useInstances = true;
        const drawWrapper = subMesh._drawWrapper;
        let defines = subMesh.materialDefines;
        if (defines && this._isDirty) {
            defines.markAsUnprocessed();
        }
        if (drawWrapper.effect && this.isFrozen) {
            if (drawWrapper._wasPreviouslyReady && drawWrapper._wasPreviouslyUsingInstances === useInstances) {
                return true;
            }
        }
        if (!subMesh.materialDefines) {
            defines = subMesh.materialDefines = new GaussianSplattingMaterialDefines();
        }
        const scene = this.getScene();
        if (this._isReadyForSubMesh(subMesh)) {
            return true;
        }
        if (!this._sourceMesh) {
            return false;
        }
        const engine = scene.getEngine();
        const gsMesh = this._sourceMesh;
        // Misc.
        PrepareDefinesForMisc(mesh, scene, this._useLogarithmicDepth, this.pointsCloud, this.fogEnabled, false, defines, undefined, undefined, undefined, this._isVertexOutputInvariant);
        // Values that need to be evaluated on every frame
        PrepareDefinesForFrameBoundValues(scene, engine, this, defines, useInstances, null, true);
        // Attribs
        PrepareDefinesForAttributes(mesh, defines, false, false);
        // SH is disabled for webGL1
        if (engine.version > 1 || engine.isWebGPU) {
            defines["SH_DEGREE"] = gsMesh.shDegree;
        }
        // Compensation
        const splatMaterial = gsMesh.material;
        defines["COMPENSATION"] = splatMaterial && splatMaterial.compensation ? splatMaterial.compensation : GaussianSplattingMaterial.Compensation;
        // Get correct effect
        if (defines.isDirty) {
            defines.markAsProcessed();
            scene.resetCachedMaterial();
            //Attributes
            PrepareAttributesForInstances(GaussianSplattingMaterial._Attribs, defines);
            PrepareUniformsAndSamplersList({
                uniformsNames: GaussianSplattingMaterial._Uniforms,
                uniformBuffersNames: GaussianSplattingMaterial._UniformBuffers,
                samplers: GaussianSplattingMaterial._Samplers,
                defines: defines,
            });
            AddClipPlaneUniforms(GaussianSplattingMaterial._Uniforms);
            const join = defines.toString();
            const effect = scene.getEngine().createEffect("gaussianSplatting", {
                attributes: GaussianSplattingMaterial._Attribs,
                uniformsNames: GaussianSplattingMaterial._Uniforms,
                uniformBuffersNames: GaussianSplattingMaterial._UniformBuffers,
                samplers: GaussianSplattingMaterial._Samplers,
                defines: join,
                onCompiled: this.onCompiled,
                onError: this.onError,
                indexParameters: {},
                shaderLanguage: this._shaderLanguage,
                extraInitializationsAsync: async () => {
                    if (this._shaderLanguage === 1 /* ShaderLanguage.WGSL */) {
                        await Promise.all([__vitePreload(() => Promise.resolve().then(() => gaussianSplatting_fragment),true              ?void 0:void 0,import.meta.url), __vitePreload(() => Promise.resolve().then(() => gaussianSplatting_vertex),true              ?void 0:void 0,import.meta.url)]);
                    }
                    else {
                        await Promise.all([__vitePreload(() => Promise.resolve().then(() => gaussianSplatting_fragment$1),true              ?void 0:void 0,import.meta.url), __vitePreload(() => Promise.resolve().then(() => gaussianSplatting_vertex$1),true              ?void 0:void 0,import.meta.url)]);
                    }
                },
            }, engine);
            subMesh.setEffect(effect, defines, this._materialContext);
        }
        if (!subMesh.effect || !subMesh.effect.isReady()) {
            return false;
        }
        defines._renderId = scene.getRenderId();
        drawWrapper._wasPreviouslyReady = true;
        drawWrapper._wasPreviouslyUsingInstances = useInstances;
        this._isDirty = false;
        return true;
    }
    /**
     * GaussianSplattingMaterial belongs to a single mesh
     * @param mesh mesh this material belongs to
     */
    setSourceMesh(mesh) {
        this._sourceMesh = mesh;
    }
    /**
     * Bind material effect for a specific Gaussian Splatting mesh
     * @param mesh Gaussian splatting mesh
     * @param effect Splatting material or node material
     * @param scene scene that contains mesh and camera used for rendering
     */
    static BindEffect(mesh, effect, scene) {
        const engine = scene.getEngine();
        const camera = scene.activeCamera;
        const renderWidth = engine.getRenderWidth() * camera.viewport.width;
        const renderHeight = engine.getRenderHeight() * camera.viewport.height;
        const gsMaterial = mesh.material;
        if (!gsMaterial._sourceMesh) {
            return;
        }
        const gsMesh = gsMaterial._sourceMesh;
        // check if rigcamera, get number of rigs
        const numberOfRigs = camera?.rigParent?.rigCameras.length || 1;
        effect.setFloat2("invViewport", 1 / (renderWidth / numberOfRigs), 1 / renderHeight);
        let focal = 1000;
        if (camera) {
            /*
            more explicit version:
            const t = camera.getProjectionMatrix().m[5];
            const FovY = Math.atan(1.0 / t) * 2.0;
            focal = renderHeight / 2.0 / Math.tan(FovY / 2.0);
            Using a shorter version here to not have tan(atan) and 2.0 factor
            */
            const t = camera.getProjectionMatrix().m[5];
            if (camera.fovMode == Camera.FOVMODE_VERTICAL_FIXED) {
                focal = (renderHeight * t) / 2.0;
            }
            else {
                focal = (renderWidth * t) / 2.0;
            }
        }
        effect.setFloat2("focal", focal, focal);
        effect.setFloat("kernelSize", gsMaterial && gsMaterial.kernelSize ? gsMaterial.kernelSize : GaussianSplattingMaterial.KernelSize);
        effect.setFloat("alpha", gsMaterial.alpha);
        scene.bindEyePosition(effect, "eyePosition", true);
        if (gsMesh.covariancesATexture) {
            const textureSize = gsMesh.covariancesATexture.getSize();
            effect.setFloat2("dataTextureSize", textureSize.width, textureSize.height);
            effect.setTexture("covariancesATexture", gsMesh.covariancesATexture);
            effect.setTexture("covariancesBTexture", gsMesh.covariancesBTexture);
            effect.setTexture("centersTexture", gsMesh.centersTexture);
            effect.setTexture("colorsTexture", gsMesh.colorsTexture);
            if (gsMesh.shTextures) {
                for (let i = 0; i < gsMesh.shTextures?.length; i++) {
                    effect.setTexture(`shTexture${i}`, gsMesh.shTextures[i]);
                }
            }
        }
    }
    /**
     * Binds the submesh to this material by preparing the effect and shader to draw
     * @param world defines the world transformation matrix
     * @param mesh defines the mesh containing the submesh
     * @param subMesh defines the submesh to bind the material to
     */
    bindForSubMesh(world, mesh, subMesh) {
        const scene = this.getScene();
        const defines = subMesh.materialDefines;
        if (!defines) {
            return;
        }
        const effect = subMesh.effect;
        if (!effect) {
            return;
        }
        this._activeEffect = effect;
        // Matrices Mesh.
        mesh.getMeshUniformBuffer().bindToEffect(effect, "Mesh");
        mesh.transferToEffect(world);
        // Bind data
        const mustRebind = this._mustRebind(scene, effect, subMesh, mesh.visibility);
        if (mustRebind) {
            this.bindView(effect);
            this.bindViewProjection(effect);
            GaussianSplattingMaterial.BindEffect(mesh, this._activeEffect, scene);
            // Clip plane
            BindClipPlane(effect, this, scene);
        }
        else if (scene.getEngine()._features.needToAlwaysBindUniformBuffers) {
            this._needToBindSceneUbo = true;
        }
        // Fog
        BindFogParameters(scene, mesh, effect);
        // Log. depth
        if (this.useLogarithmicDepth) {
            BindLogDepth(defines, effect, scene);
        }
        this._afterBind(mesh, this._activeEffect, subMesh);
    }
    static _BindEffectUniforms(gsMesh, gsMaterial, shaderMaterial, scene) {
        const engine = scene.getEngine();
        const effect = shaderMaterial.getEffect();
        gsMesh.getMeshUniformBuffer().bindToEffect(effect, "Mesh");
        shaderMaterial.bindView(effect);
        shaderMaterial.bindViewProjection(effect);
        const renderWidth = engine.getRenderWidth();
        const renderHeight = engine.getRenderHeight();
        effect.setFloat2("invViewport", 1 / renderWidth, 1 / renderHeight);
        const projection = scene.getProjectionMatrix();
        const t = projection.m[5];
        const focal = (renderWidth * t) / 2.0;
        effect.setFloat2("focal", focal, focal);
        effect.setFloat("kernelSize", gsMaterial && gsMaterial.kernelSize ? gsMaterial.kernelSize : GaussianSplattingMaterial.KernelSize);
        effect.setFloat("alpha", gsMaterial.alpha);
        let minZ, maxZ;
        const camera = scene.activeCamera;
        if (!camera) {
            return;
        }
        const cameraIsOrtho = camera.mode === Camera.ORTHOGRAPHIC_CAMERA;
        if (cameraIsOrtho) {
            minZ = !engine.useReverseDepthBuffer && engine.isNDCHalfZRange ? 0 : 1;
            maxZ = engine.useReverseDepthBuffer && engine.isNDCHalfZRange ? 0 : 1;
        }
        else {
            minZ = engine.useReverseDepthBuffer && engine.isNDCHalfZRange ? camera.minZ : engine.isNDCHalfZRange ? 0 : camera.minZ;
            maxZ = engine.useReverseDepthBuffer && engine.isNDCHalfZRange ? 0 : camera.maxZ;
        }
        effect.setFloat2("depthValues", minZ, minZ + maxZ);
        if (gsMesh.covariancesATexture) {
            const textureSize = gsMesh.covariancesATexture.getSize();
            effect.setFloat2("dataTextureSize", textureSize.width, textureSize.height);
            effect.setTexture("covariancesATexture", gsMesh.covariancesATexture);
            effect.setTexture("covariancesBTexture", gsMesh.covariancesBTexture);
            effect.setTexture("centersTexture", gsMesh.centersTexture);
            effect.setTexture("colorsTexture", gsMesh.colorsTexture);
        }
    }
    /**
     * Create a depth rendering material for a Gaussian Splatting mesh
     * @param scene scene it belongs to
     * @param shaderLanguage GLSL or WGSL
     * @returns depth rendering shader material
     */
    makeDepthRenderingMaterial(scene, shaderLanguage) {
        const shaderMaterial = new ShaderMaterial("gaussianSplattingDepthRender", scene, {
            vertex: "gaussianSplattingDepth",
            fragment: "gaussianSplattingDepth",
        }, {
            attributes: GaussianSplattingMaterial._Attribs,
            uniforms: GaussianSplattingMaterial._Uniforms,
            samplers: GaussianSplattingMaterial._Samplers,
            uniformBuffers: GaussianSplattingMaterial._UniformBuffers,
            shaderLanguage: shaderLanguage,
            defines: ["#define DEPTH_RENDER"],
        });
        shaderMaterial.onBindObservable.add((mesh) => {
            const gsMaterial = mesh.material;
            const gsMesh = mesh;
            GaussianSplattingMaterial._BindEffectUniforms(gsMesh, gsMaterial, shaderMaterial, scene);
        });
        return shaderMaterial;
    }
    static _MakeGaussianSplattingShadowDepthWrapper(scene, shaderLanguage) {
        const shaderMaterial = new ShaderMaterial("gaussianSplattingDepth", scene, {
            vertex: "gaussianSplattingDepth",
            fragment: "gaussianSplattingDepth",
        }, {
            attributes: GaussianSplattingMaterial._Attribs,
            uniforms: GaussianSplattingMaterial._Uniforms,
            samplers: GaussianSplattingMaterial._Samplers,
            uniformBuffers: GaussianSplattingMaterial._UniformBuffers,
            shaderLanguage: shaderLanguage,
        });
        const shadowDepthWrapper = new ShadowDepthWrapper(shaderMaterial, scene, {
            standalone: true,
        });
        shaderMaterial.onBindObservable.add((mesh) => {
            const gsMaterial = mesh.material;
            const gsMesh = mesh;
            GaussianSplattingMaterial._BindEffectUniforms(gsMesh, gsMaterial, shaderMaterial, scene);
        });
        return shadowDepthWrapper;
    }
    /**
     * Clones the material.
     * @param name The cloned name.
     * @returns The cloned material.
     */
    clone(name) {
        return SerializationHelper.Clone(() => new GaussianSplattingMaterial(name, this.getScene()), this);
    }
    /**
     * Serializes the current material to its JSON representation.
     * @returns The JSON representation.
     */
    serialize() {
        const serializationObject = super.serialize();
        serializationObject.customType = "BABYLON.GaussianSplattingMaterial";
        return serializationObject;
    }
    /**
     * Gets the class name of the material
     * @returns "GaussianSplattingMaterial"
     */
    getClassName() {
        return "GaussianSplattingMaterial";
    }
    /**
     * Parse a JSON input to create back a Gaussian Splatting material.
     * @param source The JSON data to parse
     * @param scene The scene to create the parsed material in
     * @param rootUrl The root url of the assets the material depends upon
     * @returns the instantiated GaussianSplattingMaterial.
     */
    static Parse(source, scene, rootUrl) {
        return SerializationHelper.Parse(() => new GaussianSplattingMaterial(source.name, scene), source, scene, rootUrl);
    }
}
/**
 * Point spread function (default 0.3). Can be overriden per GS material
 */
GaussianSplattingMaterial.KernelSize = 0.3;
/**
 * Compensation
 */
GaussianSplattingMaterial.Compensation = false;
GaussianSplattingMaterial._Attribs = [VertexBuffer.PositionKind, "splatIndex0", "splatIndex1", "splatIndex2", "splatIndex3"];
GaussianSplattingMaterial._Samplers = ["covariancesATexture", "covariancesBTexture", "centersTexture", "colorsTexture", "shTexture0", "shTexture1", "shTexture2"];
GaussianSplattingMaterial._UniformBuffers = ["Scene", "Mesh"];
GaussianSplattingMaterial._Uniforms = [
    "world",
    "view",
    "projection",
    "vFogInfos",
    "vFogColor",
    "logarithmicDepthConstant",
    "invViewport",
    "dataTextureSize",
    "focal",
    "eyePosition",
    "kernelSize",
    "alpha",
    "depthValues",
];
RegisterClass("BABYLON.GaussianSplattingMaterial", GaussianSplattingMaterial);

/* eslint-disable @typescript-eslint/naming-convention */
const HCF = HighestCommonFactor;
/**
 * Scalar computation library
 */
const Scalar = {
    ...functions,
    /**
     * Two pi constants convenient for computation.
     */
    TwoPi: Math.PI * 2,
    /**
     * Returns -1 if value is negative and +1 is value is positive.
     * @param value the value
     * @returns the value itself if it's equal to zero.
     */
    Sign: Math.sign,
    /**
     * the log2 of value.
     * @param value the value to compute log2 of
     * @returns the log2 of value.
     */
    Log2: Math.log2,
    /**
     * Returns the highest common factor of two integers.
     * @param a first parameter
     * @param b second parameter
     * @returns HCF of a and b
     */
    HCF,
};
/* eslint-enable @typescript-eslint/naming-convention */

// @internal
const UnpackUnorm = (value, bits) => {
    const t = (1 << bits) - 1;
    return (value & t) / t;
};
// @internal
const Unpack111011 = (value, result) => {
    result.x = UnpackUnorm(value >>> 21, 11);
    result.y = UnpackUnorm(value >>> 11, 10);
    result.z = UnpackUnorm(value, 11);
};
// @internal
const Unpack8888 = (value, result) => {
    result[0] = UnpackUnorm(value >>> 24, 8) * 255;
    result[1] = UnpackUnorm(value >>> 16, 8) * 255;
    result[2] = UnpackUnorm(value >>> 8, 8) * 255;
    result[3] = UnpackUnorm(value, 8) * 255;
};
// @internal
// unpack quaternion with 2,10,10,10 format (largest element, 3x10bit element)
const UnpackRot = (value, result) => {
    const norm = 1.0 / (Math.sqrt(2) * 0.5);
    const a = (UnpackUnorm(value >>> 20, 10) - 0.5) * norm;
    const b = (UnpackUnorm(value >>> 10, 10) - 0.5) * norm;
    const c = (UnpackUnorm(value, 10) - 0.5) * norm;
    const m = Math.sqrt(1.0 - (a * a + b * b + c * c));
    switch (value >>> 30) {
        case 0:
            result.set(m, a, b, c);
            break;
        case 1:
            result.set(a, m, b, c);
            break;
        case 2:
            result.set(a, b, m, c);
            break;
        case 3:
            result.set(a, b, c, m);
            break;
    }
};
/**
 * Representation of the types
 */
var PLYType;
(function (PLYType) {
    PLYType[PLYType["FLOAT"] = 0] = "FLOAT";
    PLYType[PLYType["INT"] = 1] = "INT";
    PLYType[PLYType["UINT"] = 2] = "UINT";
    PLYType[PLYType["DOUBLE"] = 3] = "DOUBLE";
    PLYType[PLYType["UCHAR"] = 4] = "UCHAR";
    PLYType[PLYType["UNDEFINED"] = 5] = "UNDEFINED";
})(PLYType || (PLYType = {}));
/**
 * Usage types of the PLY values
 */
var PLYValue;
(function (PLYValue) {
    PLYValue[PLYValue["MIN_X"] = 0] = "MIN_X";
    PLYValue[PLYValue["MIN_Y"] = 1] = "MIN_Y";
    PLYValue[PLYValue["MIN_Z"] = 2] = "MIN_Z";
    PLYValue[PLYValue["MAX_X"] = 3] = "MAX_X";
    PLYValue[PLYValue["MAX_Y"] = 4] = "MAX_Y";
    PLYValue[PLYValue["MAX_Z"] = 5] = "MAX_Z";
    PLYValue[PLYValue["MIN_SCALE_X"] = 6] = "MIN_SCALE_X";
    PLYValue[PLYValue["MIN_SCALE_Y"] = 7] = "MIN_SCALE_Y";
    PLYValue[PLYValue["MIN_SCALE_Z"] = 8] = "MIN_SCALE_Z";
    PLYValue[PLYValue["MAX_SCALE_X"] = 9] = "MAX_SCALE_X";
    PLYValue[PLYValue["MAX_SCALE_Y"] = 10] = "MAX_SCALE_Y";
    PLYValue[PLYValue["MAX_SCALE_Z"] = 11] = "MAX_SCALE_Z";
    PLYValue[PLYValue["PACKED_POSITION"] = 12] = "PACKED_POSITION";
    PLYValue[PLYValue["PACKED_ROTATION"] = 13] = "PACKED_ROTATION";
    PLYValue[PLYValue["PACKED_SCALE"] = 14] = "PACKED_SCALE";
    PLYValue[PLYValue["PACKED_COLOR"] = 15] = "PACKED_COLOR";
    PLYValue[PLYValue["X"] = 16] = "X";
    PLYValue[PLYValue["Y"] = 17] = "Y";
    PLYValue[PLYValue["Z"] = 18] = "Z";
    PLYValue[PLYValue["SCALE_0"] = 19] = "SCALE_0";
    PLYValue[PLYValue["SCALE_1"] = 20] = "SCALE_1";
    PLYValue[PLYValue["SCALE_2"] = 21] = "SCALE_2";
    PLYValue[PLYValue["DIFFUSE_RED"] = 22] = "DIFFUSE_RED";
    PLYValue[PLYValue["DIFFUSE_GREEN"] = 23] = "DIFFUSE_GREEN";
    PLYValue[PLYValue["DIFFUSE_BLUE"] = 24] = "DIFFUSE_BLUE";
    PLYValue[PLYValue["OPACITY"] = 25] = "OPACITY";
    PLYValue[PLYValue["F_DC_0"] = 26] = "F_DC_0";
    PLYValue[PLYValue["F_DC_1"] = 27] = "F_DC_1";
    PLYValue[PLYValue["F_DC_2"] = 28] = "F_DC_2";
    PLYValue[PLYValue["F_DC_3"] = 29] = "F_DC_3";
    PLYValue[PLYValue["ROT_0"] = 30] = "ROT_0";
    PLYValue[PLYValue["ROT_1"] = 31] = "ROT_1";
    PLYValue[PLYValue["ROT_2"] = 32] = "ROT_2";
    PLYValue[PLYValue["ROT_3"] = 33] = "ROT_3";
    PLYValue[PLYValue["MIN_COLOR_R"] = 34] = "MIN_COLOR_R";
    PLYValue[PLYValue["MIN_COLOR_G"] = 35] = "MIN_COLOR_G";
    PLYValue[PLYValue["MIN_COLOR_B"] = 36] = "MIN_COLOR_B";
    PLYValue[PLYValue["MAX_COLOR_R"] = 37] = "MAX_COLOR_R";
    PLYValue[PLYValue["MAX_COLOR_G"] = 38] = "MAX_COLOR_G";
    PLYValue[PLYValue["MAX_COLOR_B"] = 39] = "MAX_COLOR_B";
    PLYValue[PLYValue["SH_0"] = 40] = "SH_0";
    PLYValue[PLYValue["SH_1"] = 41] = "SH_1";
    PLYValue[PLYValue["SH_2"] = 42] = "SH_2";
    PLYValue[PLYValue["SH_3"] = 43] = "SH_3";
    PLYValue[PLYValue["SH_4"] = 44] = "SH_4";
    PLYValue[PLYValue["SH_5"] = 45] = "SH_5";
    PLYValue[PLYValue["SH_6"] = 46] = "SH_6";
    PLYValue[PLYValue["SH_7"] = 47] = "SH_7";
    PLYValue[PLYValue["SH_8"] = 48] = "SH_8";
    PLYValue[PLYValue["SH_9"] = 49] = "SH_9";
    PLYValue[PLYValue["SH_10"] = 50] = "SH_10";
    PLYValue[PLYValue["SH_11"] = 51] = "SH_11";
    PLYValue[PLYValue["SH_12"] = 52] = "SH_12";
    PLYValue[PLYValue["SH_13"] = 53] = "SH_13";
    PLYValue[PLYValue["SH_14"] = 54] = "SH_14";
    PLYValue[PLYValue["SH_15"] = 55] = "SH_15";
    PLYValue[PLYValue["SH_16"] = 56] = "SH_16";
    PLYValue[PLYValue["SH_17"] = 57] = "SH_17";
    PLYValue[PLYValue["SH_18"] = 58] = "SH_18";
    PLYValue[PLYValue["SH_19"] = 59] = "SH_19";
    PLYValue[PLYValue["SH_20"] = 60] = "SH_20";
    PLYValue[PLYValue["SH_21"] = 61] = "SH_21";
    PLYValue[PLYValue["SH_22"] = 62] = "SH_22";
    PLYValue[PLYValue["SH_23"] = 63] = "SH_23";
    PLYValue[PLYValue["SH_24"] = 64] = "SH_24";
    PLYValue[PLYValue["SH_25"] = 65] = "SH_25";
    PLYValue[PLYValue["SH_26"] = 66] = "SH_26";
    PLYValue[PLYValue["SH_27"] = 67] = "SH_27";
    PLYValue[PLYValue["SH_28"] = 68] = "SH_28";
    PLYValue[PLYValue["SH_29"] = 69] = "SH_29";
    PLYValue[PLYValue["SH_30"] = 70] = "SH_30";
    PLYValue[PLYValue["SH_31"] = 71] = "SH_31";
    PLYValue[PLYValue["SH_32"] = 72] = "SH_32";
    PLYValue[PLYValue["SH_33"] = 73] = "SH_33";
    PLYValue[PLYValue["SH_34"] = 74] = "SH_34";
    PLYValue[PLYValue["SH_35"] = 75] = "SH_35";
    PLYValue[PLYValue["SH_36"] = 76] = "SH_36";
    PLYValue[PLYValue["SH_37"] = 77] = "SH_37";
    PLYValue[PLYValue["SH_38"] = 78] = "SH_38";
    PLYValue[PLYValue["SH_39"] = 79] = "SH_39";
    PLYValue[PLYValue["SH_40"] = 80] = "SH_40";
    PLYValue[PLYValue["SH_41"] = 81] = "SH_41";
    PLYValue[PLYValue["SH_42"] = 82] = "SH_42";
    PLYValue[PLYValue["SH_43"] = 83] = "SH_43";
    PLYValue[PLYValue["SH_44"] = 84] = "SH_44";
    PLYValue[PLYValue["UNDEFINED"] = 85] = "UNDEFINED";
})(PLYValue || (PLYValue = {}));
/**
 * Class used to render a gaussian splatting mesh
 */
class GaussianSplattingMesh extends Mesh {
    /**
     * If true, disables depth sorting of the splats (default: false)
     */
    get disableDepthSort() {
        return this._disableDepthSort;
    }
    set disableDepthSort(value) {
        if (!this._disableDepthSort && value) {
            this._worker?.terminate();
            this._worker = null;
            this._disableDepthSort = true;
        }
        else if (this._disableDepthSort && !value) {
            this._disableDepthSort = false;
            this._sortIsDirty = true;
            this._instanciateWorker();
        }
    }
    /**
     * View direction factor used to compute the SH view direction in the shader.
     * @deprecated Not used anymore for SH rendering
     */
    get viewDirectionFactor() {
        return Vector3.OneReadOnly;
    }
    /**
     * SH degree. 0 = no sh (default). 1 = 3 parameters. 2 = 8 parameters. 3 = 15 parameters.
     */
    get shDegree() {
        return this._shDegree;
    }
    /**
     * Number of splats in the mesh
     */
    get splatCount() {
        return this._splatIndex?.length;
    }
    /**
     * returns the splats data array buffer that contains in order : postions (3 floats), size (3 floats), color (4 bytes), orientation quaternion (4 bytes)
     */
    get splatsData() {
        return this._splatsData;
    }
    /**
     * Gets the covariancesA texture
     */
    get covariancesATexture() {
        return this._covariancesATexture;
    }
    /**
     * Gets the covariancesB texture
     */
    get covariancesBTexture() {
        return this._covariancesBTexture;
    }
    /**
     * Gets the centers texture
     */
    get centersTexture() {
        return this._centersTexture;
    }
    /**
     * Gets the colors texture
     */
    get colorsTexture() {
        return this._colorsTexture;
    }
    /**
     * Gets the SH textures
     */
    get shTextures() {
        return this._shTextures;
    }
    /**
     * Gets the kernel size
     * Documentation and mathematical explanations here:
     * https://github.com/graphdeco-inria/gaussian-splatting/issues/294#issuecomment-1772688093
     * https://github.com/autonomousvision/mip-splatting/issues/18#issuecomment-1929388931
     */
    get kernelSize() {
        return this._material instanceof GaussianSplattingMaterial ? this._material.kernelSize : 0;
    }
    /**
     * Get the compensation state
     */
    get compensation() {
        return this._material instanceof GaussianSplattingMaterial ? this._material.compensation : false;
    }
    /**
     * set rendering material
     */
    set material(value) {
        this._material = value;
        this._material.backFaceCulling = false;
        this._material.cullBackFaces = false;
        value.resetDrawCache();
    }
    /**
     * get rendering material
     */
    get material() {
        return this._material;
    }
    static _MakeSplatGeometryForMesh(mesh) {
        const vertexData = new VertexData();
        const originPositions = [-2, -2, 0, 2, -2, 0, 2, 2, 0, -2, 2, 0];
        const originIndices = [0, 1, 2, 0, 2, 3];
        const positions = [];
        const indices = [];
        for (let i = 0; i < GaussianSplattingMesh._BatchSize; i++) {
            for (let j = 0; j < 12; j++) {
                if (j == 2 || j == 5 || j == 8 || j == 11) {
                    positions.push(i); // local splat index
                }
                else {
                    positions.push(originPositions[j]);
                }
            }
            indices.push(originIndices.map((v) => v + i * 4));
        }
        vertexData.positions = positions;
        vertexData.indices = indices.flat();
        vertexData.applyToMesh(mesh);
    }
    /**
     * Creates a new gaussian splatting mesh
     * @param name defines the name of the mesh
     * @param url defines the url to load from (optional)
     * @param scene defines the hosting scene (optional)
     * @param keepInRam keep datas in ram for editing purpose
     */
    constructor(name, url = null, scene = null, keepInRam = false) {
        super(name, scene);
        this._vertexCount = 0;
        this._worker = null;
        this._modelViewMatrix = Matrix.Identity();
        this._canPostToWorker = true;
        this._readyToDisplay = false;
        this._covariancesATexture = null;
        this._covariancesBTexture = null;
        this._centersTexture = null;
        this._colorsTexture = null;
        this._splatPositions = null;
        this._splatIndex = null;
        this._shTextures = null;
        this._splatsData = null;
        this._keepInRam = false;
        this._delayedTextureUpdate = null;
        this._useRGBACovariants = false;
        this._material = null;
        this._tmpCovariances = [0, 0, 0, 0, 0, 0];
        this._sortIsDirty = false;
        this._shDegree = 0;
        this._cameraViewInfos = new Map();
        this._disableDepthSort = false;
        this._loadingPromise = null;
        this.subMeshes = [];
        new SubMesh(0, 0, 4 * GaussianSplattingMesh._BatchSize, 0, 6 * GaussianSplattingMesh._BatchSize, this);
        this.setEnabled(false);
        // webGL2 and webGPU support for RG texture with float16 is fine. not webGL1
        this._useRGBACovariants = !this.getEngine().isWebGPU && this.getEngine().version === 1.0;
        this._keepInRam = keepInRam;
        if (url) {
            this._loadingPromise = this.loadFileAsync(url);
        }
        const gaussianSplattingMaterial = new GaussianSplattingMaterial(this.name + "_material", this._scene);
        gaussianSplattingMaterial.setSourceMesh(this);
        this._material = gaussianSplattingMaterial;
        // delete meshes created for cameras on camera removal
        this._scene.onCameraRemovedObservable.add((camera) => {
            const cameraId = camera.uniqueId;
            // delete mesh for this camera
            if (this._cameraViewInfos.has(cameraId)) {
                const cameraViewInfos = this._cameraViewInfos.get(cameraId);
                cameraViewInfos?.mesh.dispose();
                this._cameraViewInfos.delete(cameraId);
            }
        });
    }
    /**
     * Get the loading promise when loading the mesh from a URL in the constructor
     * @returns constructor loading promise or null if no URL was provided
     */
    getLoadingPromise() {
        return this._loadingPromise;
    }
    /**
     * Returns the class name
     * @returns "GaussianSplattingMesh"
     */
    getClassName() {
        return "GaussianSplattingMesh";
    }
    /**
     * Returns the total number of vertices (splats) within the mesh
     * @returns the total number of vertices
     */
    getTotalVertices() {
        return this._vertexCount;
    }
    /**
     * Is this node ready to be used/rendered
     * @param completeCheck defines if a complete check (including materials and lights) has to be done (false by default)
     * @returns true when ready
     */
    isReady(completeCheck = false) {
        if (!super.isReady(completeCheck, true)) {
            return false;
        }
        if (!this._readyToDisplay) {
            // mesh is ready when worker has done at least 1 sorting
            this._postToWorker(true);
            return false;
        }
        return true;
    }
    _getCameraDirection(camera) {
        const cameraMatrix = camera.getViewMatrix();
        this.getWorldMatrix().multiplyToRef(cameraMatrix, this._modelViewMatrix);
        // return vector used to compute distance to camera
        const localDirection = TmpVectors.Vector3[1];
        localDirection.set(this._modelViewMatrix.m[2], this._modelViewMatrix.m[6], this._modelViewMatrix.m[10]);
        localDirection.normalize();
        return localDirection;
    }
    /** @internal */
    _postToWorker(forced = false) {
        const scene = this._scene;
        const frameId = scene.getFrameId();
        // force update or at least frame update for camera is outdated
        let outdated = false;
        this._cameraViewInfos.forEach((cameraViewInfos) => {
            if (cameraViewInfos.frameIdLastUpdate !== frameId) {
                outdated = true;
            }
        });
        // array of cameras used for rendering
        const cameras = this._scene.activeCameras?.length ? this._scene.activeCameras : [this._scene.activeCamera];
        // list view infos for active cameras
        const activeViewInfos = [];
        cameras.forEach((camera) => {
            if (!camera) {
                return;
            }
            const cameraId = camera.uniqueId;
            const cameraViewInfos = this._cameraViewInfos.get(cameraId);
            if (cameraViewInfos) {
                activeViewInfos.push(cameraViewInfos);
            }
            else {
                // mesh doesn't exist yet for this camera
                const cameraMesh = new Mesh(this.name + "_cameraMesh_" + cameraId, this._scene);
                // not visible with inspector or the scene graph
                cameraMesh.reservedDataStore = { hidden: true };
                cameraMesh.setEnabled(false);
                cameraMesh.material = this.material;
                GaussianSplattingMesh._MakeSplatGeometryForMesh(cameraMesh);
                const newViewInfos = {
                    camera: camera,
                    cameraDirection: new Vector3(0, 0, 0),
                    mesh: cameraMesh,
                    frameIdLastUpdate: frameId,
                    splatIndexBufferSet: false,
                };
                activeViewInfos.push(newViewInfos);
                this._cameraViewInfos.set(cameraId, newViewInfos);
            }
        });
        // sort view infos by last updated frame id: first item is the least recently updated
        activeViewInfos.sort((a, b) => a.frameIdLastUpdate - b.frameIdLastUpdate);
        const hasSortFunction = this._worker || (_native && _native.sortSplats) || this._disableDepthSort;
        if ((forced || outdated) && hasSortFunction && (this._scene.activeCameras?.length || this._scene.activeCamera) && this._canPostToWorker) {
            // view infos sorted by least recent updated frame id
            activeViewInfos.forEach((cameraViewInfos) => {
                const camera = cameraViewInfos.camera;
                const cameraDirection = this._getCameraDirection(camera);
                const previousCameraDirection = cameraViewInfos.cameraDirection;
                const dot = Vector3.Dot(cameraDirection, previousCameraDirection);
                if ((forced || Math.abs(dot - 1) >= 0.01) && this._canPostToWorker) {
                    cameraViewInfos.cameraDirection.copyFrom(cameraDirection);
                    cameraViewInfos.frameIdLastUpdate = frameId;
                    this._canPostToWorker = false;
                    if (this._worker) {
                        this._worker.postMessage({
                            view: this._modelViewMatrix.m,
                            depthMix: this._depthMix,
                            useRightHandedSystem: this._scene.useRightHandedSystem,
                            cameraId: camera.uniqueId,
                        }, [this._depthMix.buffer]);
                    }
                    else if (_native && _native.sortSplats) {
                        _native.sortSplats(this._modelViewMatrix, this._splatPositions, this._splatIndex, this._scene.useRightHandedSystem);
                        if (cameraViewInfos.splatIndexBufferSet) {
                            cameraViewInfos.mesh.thinInstanceBufferUpdated("splatIndex");
                        }
                        else {
                            cameraViewInfos.mesh.thinInstanceSetBuffer("splatIndex", this._splatIndex, 16, false);
                            cameraViewInfos.splatIndexBufferSet = true;
                        }
                        this._canPostToWorker = true;
                        this._readyToDisplay = true;
                    }
                }
            });
        }
        else if (this._disableDepthSort) {
            activeViewInfos.forEach((cameraViewInfos) => {
                if (!cameraViewInfos.splatIndexBufferSet) {
                    cameraViewInfos.mesh.thinInstanceSetBuffer("splatIndex", this._splatIndex, 16, false);
                    cameraViewInfos.splatIndexBufferSet = true;
                }
            });
            this._canPostToWorker = true;
            this._readyToDisplay = true;
        }
    }
    /**
     * Triggers the draw call for the mesh. Usually, you don't need to call this method by your own because the mesh rendering is handled by the scene rendering manager
     * @param subMesh defines the subMesh to render
     * @param enableAlphaMode defines if alpha mode can be changed
     * @param effectiveMeshReplacement defines an optional mesh used to provide info for the rendering
     * @returns the current mesh
     */
    render(subMesh, enableAlphaMode, effectiveMeshReplacement) {
        this._postToWorker();
        // geometry used for shadows, bind the first found in the camera view infos
        if (!this._geometry && this._cameraViewInfos.size) {
            this._geometry = this._cameraViewInfos.values().next().value.mesh.geometry;
        }
        const cameraId = this._scene.activeCamera.uniqueId;
        const cameraViewInfos = this._cameraViewInfos.get(cameraId);
        if (!cameraViewInfos || !cameraViewInfos.splatIndexBufferSet) {
            return this;
        }
        const mesh = cameraViewInfos.mesh;
        mesh.getWorldMatrix().copyFrom(this.getWorldMatrix());
        return mesh.render(subMesh, enableAlphaMode, effectiveMeshReplacement);
    }
    static _TypeNameToEnum(name) {
        switch (name) {
            case "float":
                return 0 /* PLYType.FLOAT */;
            case "int":
                return 1 /* PLYType.INT */;
            case "uint":
                return 2 /* PLYType.UINT */;
            case "double":
                return 3 /* PLYType.DOUBLE */;
            case "uchar":
                return 4 /* PLYType.UCHAR */;
        }
        return 5 /* PLYType.UNDEFINED */;
    }
    static _ValueNameToEnum(name) {
        switch (name) {
            case "min_x":
                return 0 /* PLYValue.MIN_X */;
            case "min_y":
                return 1 /* PLYValue.MIN_Y */;
            case "min_z":
                return 2 /* PLYValue.MIN_Z */;
            case "max_x":
                return 3 /* PLYValue.MAX_X */;
            case "max_y":
                return 4 /* PLYValue.MAX_Y */;
            case "max_z":
                return 5 /* PLYValue.MAX_Z */;
            case "min_scale_x":
                return 6 /* PLYValue.MIN_SCALE_X */;
            case "min_scale_y":
                return 7 /* PLYValue.MIN_SCALE_Y */;
            case "min_scale_z":
                return 8 /* PLYValue.MIN_SCALE_Z */;
            case "max_scale_x":
                return 9 /* PLYValue.MAX_SCALE_X */;
            case "max_scale_y":
                return 10 /* PLYValue.MAX_SCALE_Y */;
            case "max_scale_z":
                return 11 /* PLYValue.MAX_SCALE_Z */;
            case "packed_position":
                return 12 /* PLYValue.PACKED_POSITION */;
            case "packed_rotation":
                return 13 /* PLYValue.PACKED_ROTATION */;
            case "packed_scale":
                return 14 /* PLYValue.PACKED_SCALE */;
            case "packed_color":
                return 15 /* PLYValue.PACKED_COLOR */;
            case "x":
                return 16 /* PLYValue.X */;
            case "y":
                return 17 /* PLYValue.Y */;
            case "z":
                return 18 /* PLYValue.Z */;
            case "scale_0":
                return 19 /* PLYValue.SCALE_0 */;
            case "scale_1":
                return 20 /* PLYValue.SCALE_1 */;
            case "scale_2":
                return 21 /* PLYValue.SCALE_2 */;
            case "diffuse_red":
            case "red":
                return 22 /* PLYValue.DIFFUSE_RED */;
            case "diffuse_green":
            case "green":
                return 23 /* PLYValue.DIFFUSE_GREEN */;
            case "diffuse_blue":
            case "blue":
                return 24 /* PLYValue.DIFFUSE_BLUE */;
            case "f_dc_0":
                return 26 /* PLYValue.F_DC_0 */;
            case "f_dc_1":
                return 27 /* PLYValue.F_DC_1 */;
            case "f_dc_2":
                return 28 /* PLYValue.F_DC_2 */;
            case "f_dc_3":
                return 29 /* PLYValue.F_DC_3 */;
            case "opacity":
                return 25 /* PLYValue.OPACITY */;
            case "rot_0":
                return 30 /* PLYValue.ROT_0 */;
            case "rot_1":
                return 31 /* PLYValue.ROT_1 */;
            case "rot_2":
                return 32 /* PLYValue.ROT_2 */;
            case "rot_3":
                return 33 /* PLYValue.ROT_3 */;
            case "min_r":
                return 34 /* PLYValue.MIN_COLOR_R */;
            case "min_g":
                return 35 /* PLYValue.MIN_COLOR_G */;
            case "min_b":
                return 36 /* PLYValue.MIN_COLOR_B */;
            case "max_r":
                return 37 /* PLYValue.MAX_COLOR_R */;
            case "max_g":
                return 38 /* PLYValue.MAX_COLOR_G */;
            case "max_b":
                return 39 /* PLYValue.MAX_COLOR_B */;
            case "f_rest_0":
                return 40 /* PLYValue.SH_0 */;
            case "f_rest_1":
                return 41 /* PLYValue.SH_1 */;
            case "f_rest_2":
                return 42 /* PLYValue.SH_2 */;
            case "f_rest_3":
                return 43 /* PLYValue.SH_3 */;
            case "f_rest_4":
                return 44 /* PLYValue.SH_4 */;
            case "f_rest_5":
                return 45 /* PLYValue.SH_5 */;
            case "f_rest_6":
                return 46 /* PLYValue.SH_6 */;
            case "f_rest_7":
                return 47 /* PLYValue.SH_7 */;
            case "f_rest_8":
                return 48 /* PLYValue.SH_8 */;
            case "f_rest_9":
                return 49 /* PLYValue.SH_9 */;
            case "f_rest_10":
                return 50 /* PLYValue.SH_10 */;
            case "f_rest_11":
                return 51 /* PLYValue.SH_11 */;
            case "f_rest_12":
                return 52 /* PLYValue.SH_12 */;
            case "f_rest_13":
                return 53 /* PLYValue.SH_13 */;
            case "f_rest_14":
                return 54 /* PLYValue.SH_14 */;
            case "f_rest_15":
                return 55 /* PLYValue.SH_15 */;
            case "f_rest_16":
                return 56 /* PLYValue.SH_16 */;
            case "f_rest_17":
                return 57 /* PLYValue.SH_17 */;
            case "f_rest_18":
                return 58 /* PLYValue.SH_18 */;
            case "f_rest_19":
                return 59 /* PLYValue.SH_19 */;
            case "f_rest_20":
                return 60 /* PLYValue.SH_20 */;
            case "f_rest_21":
                return 61 /* PLYValue.SH_21 */;
            case "f_rest_22":
                return 62 /* PLYValue.SH_22 */;
            case "f_rest_23":
                return 63 /* PLYValue.SH_23 */;
            case "f_rest_24":
                return 64 /* PLYValue.SH_24 */;
            case "f_rest_25":
                return 65 /* PLYValue.SH_25 */;
            case "f_rest_26":
                return 66 /* PLYValue.SH_26 */;
            case "f_rest_27":
                return 67 /* PLYValue.SH_27 */;
            case "f_rest_28":
                return 68 /* PLYValue.SH_28 */;
            case "f_rest_29":
                return 69 /* PLYValue.SH_29 */;
            case "f_rest_30":
                return 70 /* PLYValue.SH_30 */;
            case "f_rest_31":
                return 71 /* PLYValue.SH_31 */;
            case "f_rest_32":
                return 72 /* PLYValue.SH_32 */;
            case "f_rest_33":
                return 73 /* PLYValue.SH_33 */;
            case "f_rest_34":
                return 74 /* PLYValue.SH_34 */;
            case "f_rest_35":
                return 75 /* PLYValue.SH_35 */;
            case "f_rest_36":
                return 76 /* PLYValue.SH_36 */;
            case "f_rest_37":
                return 77 /* PLYValue.SH_37 */;
            case "f_rest_38":
                return 78 /* PLYValue.SH_38 */;
            case "f_rest_39":
                return 79 /* PLYValue.SH_39 */;
            case "f_rest_40":
                return 80 /* PLYValue.SH_40 */;
            case "f_rest_41":
                return 81 /* PLYValue.SH_41 */;
            case "f_rest_42":
                return 82 /* PLYValue.SH_42 */;
            case "f_rest_43":
                return 83 /* PLYValue.SH_43 */;
            case "f_rest_44":
                return 84 /* PLYValue.SH_44 */;
        }
        return 85 /* PLYValue.UNDEFINED */;
    }
    /**
     * Parse a PLY file header and returns metas infos on splats and chunks
     * @param data the loaded buffer
     * @returns a PLYHeader
     */
    static ParseHeader(data) {
        const ubuf = new Uint8Array(data);
        const header = new TextDecoder().decode(ubuf.slice(0, 1024 * 10));
        const headerEnd = "end_header\n";
        const headerEndIndex = header.indexOf(headerEnd);
        if (headerEndIndex < 0 || !header) {
            // standard splat
            return null;
        }
        const vertexCount = parseInt(/element vertex (\d+)\n/.exec(header)[1]);
        const chunkElement = /element chunk (\d+)\n/.exec(header);
        let chunkCount = 0;
        if (chunkElement) {
            chunkCount = parseInt(chunkElement[1]);
        }
        let rowVertexOffset = 0;
        let rowChunkOffset = 0;
        const offsets = {
            double: 8,
            int: 4,
            uint: 4,
            float: 4,
            short: 2,
            ushort: 2,
            uchar: 1,
            list: 0,
        };
        let ElementMode;
        (function (ElementMode) {
            ElementMode[ElementMode["Vertex"] = 0] = "Vertex";
            ElementMode[ElementMode["Chunk"] = 1] = "Chunk";
            ElementMode[ElementMode["SH"] = 2] = "SH";
            ElementMode[ElementMode["Unused"] = 3] = "Unused";
        })(ElementMode || (ElementMode = {}));
        let chunkMode = 1 /* ElementMode.Chunk */;
        const vertexProperties = [];
        const chunkProperties = [];
        const filtered = header.slice(0, headerEndIndex).split("\n");
        let shDegree = 0;
        for (const prop of filtered) {
            if (prop.startsWith("property ")) {
                const [, typeName, name] = prop.split(" ");
                const value = GaussianSplattingMesh._ValueNameToEnum(name);
                if (value != 85 /* PLYValue.UNDEFINED */) {
                    // SH degree 1,2 or 3 for 9, 24 or 45 values
                    if (value >= 84 /* PLYValue.SH_44 */) {
                        shDegree = 3;
                    }
                    else if (value >= 64 /* PLYValue.SH_24 */) {
                        shDegree = Math.max(shDegree, 2);
                    }
                    else if (value >= 48 /* PLYValue.SH_8 */) {
                        shDegree = Math.max(shDegree, 1);
                    }
                }
                const type = GaussianSplattingMesh._TypeNameToEnum(typeName);
                if (chunkMode == 1 /* ElementMode.Chunk */) {
                    chunkProperties.push({ value, type, offset: rowChunkOffset });
                    rowChunkOffset += offsets[typeName];
                }
                else if (chunkMode == 0 /* ElementMode.Vertex */) {
                    vertexProperties.push({ value, type, offset: rowVertexOffset });
                    rowVertexOffset += offsets[typeName];
                }
                else if (chunkMode == 2 /* ElementMode.SH */) {
                    // SH doesn't count for vertex row size but its properties are used to retrieve SH
                    vertexProperties.push({ value, type, offset: rowVertexOffset });
                }
                if (!offsets[typeName]) {
                    Logger.Warn(`Unsupported property type: ${typeName}.`);
                }
            }
            else if (prop.startsWith("element ")) {
                const [, type] = prop.split(" ");
                if (type == "chunk") {
                    chunkMode = 1 /* ElementMode.Chunk */;
                }
                else if (type == "vertex") {
                    chunkMode = 0 /* ElementMode.Vertex */;
                }
                else if (type == "sh") {
                    chunkMode = 2 /* ElementMode.SH */;
                }
                else {
                    chunkMode = 3 /* ElementMode.Unused */;
                }
            }
        }
        const dataView = new DataView(data, headerEndIndex + headerEnd.length);
        const buffer = new ArrayBuffer(GaussianSplattingMesh._RowOutputLength * vertexCount);
        let shBuffer = null;
        let shCoefficientCount = 0;
        if (shDegree) {
            const shVectorCount = (shDegree + 1) * (shDegree + 1) - 1;
            shCoefficientCount = shVectorCount * 3;
            shBuffer = new ArrayBuffer(shCoefficientCount * vertexCount);
        }
        return {
            vertexCount: vertexCount,
            chunkCount: chunkCount,
            rowVertexLength: rowVertexOffset,
            rowChunkLength: rowChunkOffset,
            vertexProperties: vertexProperties,
            chunkProperties: chunkProperties,
            dataView: dataView,
            buffer: buffer,
            shDegree: shDegree,
            shCoefficientCount: shCoefficientCount,
            shBuffer: shBuffer,
        };
    }
    static _GetCompressedChunks(header, offset) {
        if (!header.chunkCount) {
            return null;
        }
        const dataView = header.dataView;
        const compressedChunks = new Array(header.chunkCount);
        for (let i = 0; i < header.chunkCount; i++) {
            const currentChunk = {
                min: new Vector3(),
                max: new Vector3(),
                minScale: new Vector3(),
                maxScale: new Vector3(),
                minColor: new Vector3(0, 0, 0),
                maxColor: new Vector3(1, 1, 1),
            };
            compressedChunks[i] = currentChunk;
            for (let propertyIndex = 0; propertyIndex < header.chunkProperties.length; propertyIndex++) {
                const property = header.chunkProperties[propertyIndex];
                let value;
                switch (property.type) {
                    case 0 /* PLYType.FLOAT */:
                        value = dataView.getFloat32(property.offset + offset.value, true);
                        break;
                    default:
                        continue;
                }
                switch (property.value) {
                    case 0 /* PLYValue.MIN_X */:
                        currentChunk.min.x = value;
                        break;
                    case 1 /* PLYValue.MIN_Y */:
                        currentChunk.min.y = value;
                        break;
                    case 2 /* PLYValue.MIN_Z */:
                        currentChunk.min.z = value;
                        break;
                    case 3 /* PLYValue.MAX_X */:
                        currentChunk.max.x = value;
                        break;
                    case 4 /* PLYValue.MAX_Y */:
                        currentChunk.max.y = value;
                        break;
                    case 5 /* PLYValue.MAX_Z */:
                        currentChunk.max.z = value;
                        break;
                    case 6 /* PLYValue.MIN_SCALE_X */:
                        currentChunk.minScale.x = value;
                        break;
                    case 7 /* PLYValue.MIN_SCALE_Y */:
                        currentChunk.minScale.y = value;
                        break;
                    case 8 /* PLYValue.MIN_SCALE_Z */:
                        currentChunk.minScale.z = value;
                        break;
                    case 9 /* PLYValue.MAX_SCALE_X */:
                        currentChunk.maxScale.x = value;
                        break;
                    case 10 /* PLYValue.MAX_SCALE_Y */:
                        currentChunk.maxScale.y = value;
                        break;
                    case 11 /* PLYValue.MAX_SCALE_Z */:
                        currentChunk.maxScale.z = value;
                        break;
                    case 34 /* PLYValue.MIN_COLOR_R */:
                        currentChunk.minColor.x = value;
                        break;
                    case 35 /* PLYValue.MIN_COLOR_G */:
                        currentChunk.minColor.y = value;
                        break;
                    case 36 /* PLYValue.MIN_COLOR_B */:
                        currentChunk.minColor.z = value;
                        break;
                    case 37 /* PLYValue.MAX_COLOR_R */:
                        currentChunk.maxColor.x = value;
                        break;
                    case 38 /* PLYValue.MAX_COLOR_G */:
                        currentChunk.maxColor.y = value;
                        break;
                    case 39 /* PLYValue.MAX_COLOR_B */:
                        currentChunk.maxColor.z = value;
                        break;
                }
            }
            offset.value += header.rowChunkLength;
        }
        return compressedChunks;
    }
    static _GetSplat(header, index, compressedChunks, offset) {
        const q = TmpVectors.Quaternion[0];
        const temp3 = TmpVectors.Vector3[0];
        const rowOutputLength = GaussianSplattingMesh._RowOutputLength;
        const buffer = header.buffer;
        const dataView = header.dataView;
        const position = new Float32Array(buffer, index * rowOutputLength, 3);
        const scale = new Float32Array(buffer, index * rowOutputLength + 12, 3);
        const rgba = new Uint8ClampedArray(buffer, index * rowOutputLength + 24, 4);
        const rot = new Uint8ClampedArray(buffer, index * rowOutputLength + 28, 4);
        let sh = null;
        if (header.shBuffer) {
            sh = new Uint8ClampedArray(header.shBuffer, index * header.shCoefficientCount, header.shCoefficientCount);
        }
        const chunkIndex = index >> 8;
        let r0 = 255;
        let r1 = 0;
        let r2 = 0;
        let r3 = 0;
        const plySH = [];
        for (let propertyIndex = 0; propertyIndex < header.vertexProperties.length; propertyIndex++) {
            const property = header.vertexProperties[propertyIndex];
            let value;
            switch (property.type) {
                case 0 /* PLYType.FLOAT */:
                    value = dataView.getFloat32(offset.value + property.offset, true);
                    break;
                case 1 /* PLYType.INT */:
                    value = dataView.getInt32(offset.value + property.offset, true);
                    break;
                case 2 /* PLYType.UINT */:
                    value = dataView.getUint32(offset.value + property.offset, true);
                    break;
                case 3 /* PLYType.DOUBLE */:
                    value = dataView.getFloat64(offset.value + property.offset, true);
                    break;
                case 4 /* PLYType.UCHAR */:
                    value = dataView.getUint8(offset.value + property.offset);
                    break;
                default:
                    continue;
            }
            switch (property.value) {
                case 12 /* PLYValue.PACKED_POSITION */:
                    {
                        const compressedChunk = compressedChunks[chunkIndex];
                        Unpack111011(value, temp3);
                        position[0] = Scalar.Lerp(compressedChunk.min.x, compressedChunk.max.x, temp3.x);
                        position[1] = Scalar.Lerp(compressedChunk.min.y, compressedChunk.max.y, temp3.y);
                        position[2] = Scalar.Lerp(compressedChunk.min.z, compressedChunk.max.z, temp3.z);
                    }
                    break;
                case 13 /* PLYValue.PACKED_ROTATION */:
                    {
                        UnpackRot(value, q);
                        r0 = q.x;
                        r1 = q.y;
                        r2 = q.z;
                        r3 = q.w;
                    }
                    break;
                case 14 /* PLYValue.PACKED_SCALE */:
                    {
                        const compressedChunk = compressedChunks[chunkIndex];
                        Unpack111011(value, temp3);
                        scale[0] = Math.exp(Scalar.Lerp(compressedChunk.minScale.x, compressedChunk.maxScale.x, temp3.x));
                        scale[1] = Math.exp(Scalar.Lerp(compressedChunk.minScale.y, compressedChunk.maxScale.y, temp3.y));
                        scale[2] = Math.exp(Scalar.Lerp(compressedChunk.minScale.z, compressedChunk.maxScale.z, temp3.z));
                    }
                    break;
                case 15 /* PLYValue.PACKED_COLOR */:
                    {
                        const compressedChunk = compressedChunks[chunkIndex];
                        Unpack8888(value, rgba);
                        rgba[0] = Scalar.Lerp(compressedChunk.minColor.x, compressedChunk.maxColor.x, rgba[0] / 255) * 255;
                        rgba[1] = Scalar.Lerp(compressedChunk.minColor.y, compressedChunk.maxColor.y, rgba[1] / 255) * 255;
                        rgba[2] = Scalar.Lerp(compressedChunk.minColor.z, compressedChunk.maxColor.z, rgba[2] / 255) * 255;
                    }
                    break;
                case 16 /* PLYValue.X */:
                    position[0] = value;
                    break;
                case 17 /* PLYValue.Y */:
                    position[1] = value;
                    break;
                case 18 /* PLYValue.Z */:
                    position[2] = value;
                    break;
                case 19 /* PLYValue.SCALE_0 */:
                    scale[0] = Math.exp(value);
                    break;
                case 20 /* PLYValue.SCALE_1 */:
                    scale[1] = Math.exp(value);
                    break;
                case 21 /* PLYValue.SCALE_2 */:
                    scale[2] = Math.exp(value);
                    break;
                case 22 /* PLYValue.DIFFUSE_RED */:
                    rgba[0] = value;
                    break;
                case 23 /* PLYValue.DIFFUSE_GREEN */:
                    rgba[1] = value;
                    break;
                case 24 /* PLYValue.DIFFUSE_BLUE */:
                    rgba[2] = value;
                    break;
                case 26 /* PLYValue.F_DC_0 */:
                    rgba[0] = (0.5 + GaussianSplattingMesh._SH_C0 * value) * 255;
                    break;
                case 27 /* PLYValue.F_DC_1 */:
                    rgba[1] = (0.5 + GaussianSplattingMesh._SH_C0 * value) * 255;
                    break;
                case 28 /* PLYValue.F_DC_2 */:
                    rgba[2] = (0.5 + GaussianSplattingMesh._SH_C0 * value) * 255;
                    break;
                case 29 /* PLYValue.F_DC_3 */:
                    rgba[3] = (0.5 + GaussianSplattingMesh._SH_C0 * value) * 255;
                    break;
                case 25 /* PLYValue.OPACITY */:
                    rgba[3] = (1 / (1 + Math.exp(-value))) * 255;
                    break;
                case 30 /* PLYValue.ROT_0 */:
                    r0 = value;
                    break;
                case 31 /* PLYValue.ROT_1 */:
                    r1 = value;
                    break;
                case 32 /* PLYValue.ROT_2 */:
                    r2 = value;
                    break;
                case 33 /* PLYValue.ROT_3 */:
                    r3 = value;
                    break;
            }
            if (sh && property.value >= 40 /* PLYValue.SH_0 */ && property.value <= 84 /* PLYValue.SH_44 */) {
                const shIndex = property.value - 40 /* PLYValue.SH_0 */;
                if (property.type == 4 /* PLYType.UCHAR */ && header.chunkCount) {
                    // compressed ply. dataView points to beginning of vertex
                    // could be improved with a direct copy instead of a per SH index computation + copy
                    const compressedValue = dataView.getUint8(header.rowChunkLength * header.chunkCount + header.vertexCount * header.rowVertexLength + index * header.shCoefficientCount + shIndex);
                    // compressed .ply SH import : https://github.com/playcanvas/engine/blob/fda3f0368b45d7381f0b5a1722bd2056128eaebe/src/scene/gsplat/gsplat-compressed-data.js#L88C81-L88C98
                    plySH[shIndex] = (compressedValue * (8 / 255) - 4) * 127.5 + 127.5;
                }
                else {
                    const clampedValue = Scalar.Clamp(value * 127.5 + 127.5, 0, 255);
                    plySH[shIndex] = clampedValue;
                }
            }
        }
        if (sh) {
            const shDim = header.shDegree == 1 ? 3 : header.shDegree == 2 ? 8 : 15;
            for (let j = 0; j < shDim; j++) {
                sh[j * 3 + 0] = plySH[j];
                sh[j * 3 + 1] = plySH[j + shDim];
                sh[j * 3 + 2] = plySH[j + shDim * 2];
            }
        }
        q.set(r1, r2, r3, r0);
        q.normalize();
        rot[0] = q.w * 127.5 + 127.5;
        rot[1] = q.x * 127.5 + 127.5;
        rot[2] = q.y * 127.5 + 127.5;
        rot[3] = q.z * 127.5 + 127.5;
        offset.value += header.rowVertexLength;
    }
    /**
     * Converts a .ply data with SH coefficients splat
     * if data array buffer is not ply, returns the original buffer
     * @param data the .ply data to load
     * @param useCoroutine use coroutine and yield
     * @returns the loaded splat buffer and optional array of sh coefficients
     */
    static *ConvertPLYWithSHToSplat(data, useCoroutine = false) {
        const header = GaussianSplattingMesh.ParseHeader(data);
        if (!header) {
            return { buffer: data };
        }
        const offset = { value: 0 };
        const compressedChunks = GaussianSplattingMesh._GetCompressedChunks(header, offset);
        for (let i = 0; i < header.vertexCount; i++) {
            GaussianSplattingMesh._GetSplat(header, i, compressedChunks, offset);
            if (i % GaussianSplattingMesh._PlyConversionBatchSize === 0 && useCoroutine) {
                yield;
            }
        }
        let sh = null;
        // make SH texture buffers
        if (header.shDegree && header.shBuffer) {
            const textureCount = Math.ceil(header.shCoefficientCount / 16); // 4 components can be stored per texture, 4 sh per component
            let shIndexRead = 0;
            const ubuf = new Uint8Array(header.shBuffer);
            // sh is an array of uint8array that will be used to create sh textures
            sh = [];
            const splatCount = header.vertexCount;
            const engine = EngineStore.LastCreatedEngine;
            if (engine) {
                const width = engine.getCaps().maxTextureSize;
                const height = Math.ceil(splatCount / width);
                // create array for the number of textures needed.
                for (let textureIndex = 0; textureIndex < textureCount; textureIndex++) {
                    const texture = new Uint8Array(height * width * 4 * 4); // 4 components per texture, 4 sh per component
                    sh.push(texture);
                }
                for (let i = 0; i < splatCount; i++) {
                    for (let shIndexWrite = 0; shIndexWrite < header.shCoefficientCount; shIndexWrite++) {
                        const shValue = ubuf[shIndexRead++];
                        const textureIndex = Math.floor(shIndexWrite / 16);
                        const shArray = sh[textureIndex];
                        const byteIndexInTexture = shIndexWrite % 16; // [0..15]
                        const offsetPerSplat = i * 16; // 16 sh values per texture per splat.
                        shArray[byteIndexInTexture + offsetPerSplat] = shValue;
                    }
                }
            }
        }
        return { buffer: header.buffer, sh: sh };
    }
    /**
     * Converts a .ply data array buffer to splat
     * if data array buffer is not ply, returns the original buffer
     * @param data the .ply data to load
     * @param useCoroutine use coroutine and yield
     * @returns the loaded splat buffer without SH coefficient, whether ply contains or not SH.
     */
    static *ConvertPLYToSplat(data, useCoroutine = false) {
        const header = GaussianSplattingMesh.ParseHeader(data);
        if (!header) {
            return data;
        }
        const offset = { value: 0 };
        const compressedChunks = GaussianSplattingMesh._GetCompressedChunks(header, offset);
        for (let i = 0; i < header.vertexCount; i++) {
            GaussianSplattingMesh._GetSplat(header, i, compressedChunks, offset);
            if (i % GaussianSplattingMesh._PlyConversionBatchSize === 0 && useCoroutine) {
                yield;
            }
        }
        return header.buffer;
    }
    /**
     * Converts a .ply data array buffer to splat
     * if data array buffer is not ply, returns the original buffer
     * @param data the .ply data to load
     * @returns the loaded splat buffer
     */
    static async ConvertPLYToSplatAsync(data) {
        return await runCoroutineAsync(GaussianSplattingMesh.ConvertPLYToSplat(data, true), createYieldingScheduler());
    }
    /**
     * Converts a .ply with SH data array buffer to splat
     * if data array buffer is not ply, returns the original buffer
     * @param data the .ply data to load
     * @returns the loaded splat buffer with SH
     */
    static async ConvertPLYWithSHToSplatAsync(data) {
        return await runCoroutineAsync(GaussianSplattingMesh.ConvertPLYWithSHToSplat(data, true), createYieldingScheduler());
    }
    /**
     * Loads a .splat Gaussian Splatting array buffer asynchronously
     * @param data arraybuffer containing splat file
     * @returns a promise that resolves when the operation is complete
     */
    async loadDataAsync(data) {
        return await this.updateDataAsync(data);
    }
    /**
     * Loads a Gaussian or Splatting file asynchronously
     * @param url path to the splat file to load
     * @param scene optional scene it belongs to
     * @returns a promise that resolves when the operation is complete
     * @deprecated Please use SceneLoader.ImportMeshAsync instead
     */
    async loadFileAsync(url, scene) {
        await ImportMeshAsync(url, (scene || EngineStore.LastCreatedScene), { pluginOptions: { splat: { gaussianSplattingMesh: this } } });
    }
    /**
     * Releases resources associated with this mesh.
     * @param doNotRecurse Set to true to not recurse into each children (recurse into each children by default)
     */
    dispose(doNotRecurse) {
        this._covariancesATexture?.dispose();
        this._covariancesBTexture?.dispose();
        this._centersTexture?.dispose();
        this._colorsTexture?.dispose();
        if (this._shTextures) {
            for (const shTexture of this._shTextures) {
                shTexture.dispose();
            }
        }
        this._covariancesATexture = null;
        this._covariancesBTexture = null;
        this._centersTexture = null;
        this._colorsTexture = null;
        this._shTextures = null;
        this._worker?.terminate();
        this._worker = null;
        // delete meshes created for each camera
        this._cameraViewInfos.forEach((cameraViewInfo) => {
            cameraViewInfo.mesh.dispose();
        });
        super.dispose(doNotRecurse, true);
    }
    _copyTextures(source) {
        this._covariancesATexture = source.covariancesATexture?.clone();
        this._covariancesBTexture = source.covariancesBTexture?.clone();
        this._centersTexture = source.centersTexture?.clone();
        this._colorsTexture = source.colorsTexture?.clone();
        if (source._shTextures) {
            this._shTextures = [];
            for (const shTexture of this._shTextures) {
                this._shTextures?.push(shTexture.clone());
            }
        }
    }
    /**
     * Returns a new Mesh object generated from the current mesh properties.
     * @param name is a string, the name given to the new mesh
     * @returns a new Gaussian Splatting Mesh
     */
    clone(name = "") {
        const newGS = new GaussianSplattingMesh(name, undefined, this.getScene());
        newGS._copySource(this);
        newGS.makeGeometryUnique();
        newGS._vertexCount = this._vertexCount;
        newGS._copyTextures(this);
        newGS._modelViewMatrix = Matrix.Identity();
        newGS._splatPositions = this._splatPositions;
        newGS._readyToDisplay = false;
        newGS._disableDepthSort = this._disableDepthSort;
        newGS._instanciateWorker();
        const binfo = this.getBoundingInfo();
        newGS.getBoundingInfo().reConstruct(binfo.minimum, binfo.maximum, this.getWorldMatrix());
        newGS.forcedInstanceCount = this.forcedInstanceCount;
        newGS.setEnabled(true);
        return newGS;
    }
    _makeEmptySplat(index, covA, covB, colorArray) {
        const covBSItemSize = this._useRGBACovariants ? 4 : 2;
        this._splatPositions[4 * index + 0] = 0;
        this._splatPositions[4 * index + 1] = 0;
        this._splatPositions[4 * index + 2] = 0;
        covA[index * 4 + 0] = ToHalfFloat(0);
        covA[index * 4 + 1] = ToHalfFloat(0);
        covA[index * 4 + 2] = ToHalfFloat(0);
        covA[index * 4 + 3] = ToHalfFloat(0);
        covB[index * covBSItemSize + 0] = ToHalfFloat(0);
        covB[index * covBSItemSize + 1] = ToHalfFloat(0);
        colorArray[index * 4 + 3] = 0;
    }
    _makeSplat(index, fBuffer, uBuffer, covA, covB, colorArray, minimum, maximum, options) {
        const matrixRotation = TmpVectors.Matrix[0];
        const matrixScale = TmpVectors.Matrix[1];
        const quaternion = TmpVectors.Quaternion[0];
        const covBSItemSize = this._useRGBACovariants ? 4 : 2;
        const x = fBuffer[8 * index + 0];
        const y = fBuffer[8 * index + 1] * (options.flipY ? -1 : 1);
        const z = fBuffer[8 * index + 2];
        this._splatPositions[4 * index + 0] = x;
        this._splatPositions[4 * index + 1] = y;
        this._splatPositions[4 * index + 2] = z;
        minimum.minimizeInPlaceFromFloats(x, y, z);
        maximum.maximizeInPlaceFromFloats(x, y, z);
        quaternion.set((uBuffer[32 * index + 28 + 1] - 127.5) / 127.5, (uBuffer[32 * index + 28 + 2] - 127.5) / 127.5, (uBuffer[32 * index + 28 + 3] - 127.5) / 127.5, -(uBuffer[32 * index + 28 + 0] - 127.5) / 127.5);
        quaternion.normalize();
        quaternion.toRotationMatrix(matrixRotation);
        Matrix.ScalingToRef(fBuffer[8 * index + 3 + 0] * 2, fBuffer[8 * index + 3 + 1] * 2, fBuffer[8 * index + 3 + 2] * 2, matrixScale);
        const m = matrixRotation.multiplyToRef(matrixScale, TmpVectors.Matrix[0]).m;
        const covariances = this._tmpCovariances;
        covariances[0] = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
        covariances[1] = m[0] * m[4] + m[1] * m[5] + m[2] * m[6];
        covariances[2] = m[0] * m[8] + m[1] * m[9] + m[2] * m[10];
        covariances[3] = m[4] * m[4] + m[5] * m[5] + m[6] * m[6];
        covariances[4] = m[4] * m[8] + m[5] * m[9] + m[6] * m[10];
        covariances[5] = m[8] * m[8] + m[9] * m[9] + m[10] * m[10];
        // normalize covA, covB
        let factor = -1e4;
        for (let covIndex = 0; covIndex < 6; covIndex++) {
            factor = Math.max(factor, Math.abs(covariances[covIndex]));
        }
        this._splatPositions[4 * index + 3] = factor;
        const transform = factor;
        covA[index * 4 + 0] = ToHalfFloat(covariances[0] / transform);
        covA[index * 4 + 1] = ToHalfFloat(covariances[1] / transform);
        covA[index * 4 + 2] = ToHalfFloat(covariances[2] / transform);
        covA[index * 4 + 3] = ToHalfFloat(covariances[3] / transform);
        covB[index * covBSItemSize + 0] = ToHalfFloat(covariances[4] / transform);
        covB[index * covBSItemSize + 1] = ToHalfFloat(covariances[5] / transform);
        // colors
        colorArray[index * 4 + 0] = uBuffer[32 * index + 24 + 0];
        colorArray[index * 4 + 1] = uBuffer[32 * index + 24 + 1];
        colorArray[index * 4 + 2] = uBuffer[32 * index + 24 + 2];
        colorArray[index * 4 + 3] = uBuffer[32 * index + 24 + 3];
    }
    _updateTextures(covA, covB, colorArray, sh) {
        const textureSize = this._getTextureSize(this._vertexCount);
        // Update the textures
        const createTextureFromData = (data, width, height, format) => {
            return new RawTexture(data, width, height, format, this._scene, false, false, 2, 1);
        };
        const createTextureFromDataU8 = (data, width, height, format) => {
            return new RawTexture(data, width, height, format, this._scene, false, false, 2, 0);
        };
        const createTextureFromDataU32 = (data, width, height, format) => {
            return new RawTexture(data, width, height, format, this._scene, false, false, 1, 7);
        };
        const createTextureFromDataF16 = (data, width, height, format) => {
            return new RawTexture(data, width, height, format, this._scene, false, false, 2, 2);
        };
        if (this._covariancesATexture) {
            this._delayedTextureUpdate = { covA: covA, covB: covB, colors: colorArray, centers: this._splatPositions, sh: sh };
            const positions = Float32Array.from(this._splatPositions);
            const vertexCount = this._vertexCount;
            if (this._worker) {
                this._worker.postMessage({ positions, vertexCount }, [positions.buffer]);
            }
            this._postToWorker(true);
        }
        else {
            this._covariancesATexture = createTextureFromDataF16(covA, textureSize.x, textureSize.y, 5);
            this._covariancesBTexture = createTextureFromDataF16(covB, textureSize.x, textureSize.y, this._useRGBACovariants ? 5 : 7);
            this._centersTexture = createTextureFromData(this._splatPositions, textureSize.x, textureSize.y, 5);
            this._colorsTexture = createTextureFromDataU8(colorArray, textureSize.x, textureSize.y, 5);
            if (sh) {
                this._shTextures = [];
                for (const shData of sh) {
                    const buffer = new Uint32Array(shData.buffer);
                    const shTexture = createTextureFromDataU32(buffer, textureSize.x, textureSize.y, 11);
                    shTexture.wrapU = 0;
                    shTexture.wrapV = 0;
                    this._shTextures.push(shTexture);
                }
            }
            this._instanciateWorker();
        }
    }
    *_updateData(data, isAsync, sh, options = { flipY: false }) {
        // if a covariance texture is present, then it's not a creation but an update
        if (!this._covariancesATexture) {
            this._readyToDisplay = false;
        }
        // Parse the data
        const uBuffer = new Uint8Array(data);
        const fBuffer = new Float32Array(uBuffer.buffer);
        if (this._keepInRam) {
            this._splatsData = data;
            // keep sh in ram too ?
        }
        const vertexCount = uBuffer.length / GaussianSplattingMesh._RowOutputLength;
        if (vertexCount != this._vertexCount) {
            this._updateSplatIndexBuffer(vertexCount);
        }
        this._vertexCount = vertexCount;
        // degree == 1 for 1 texture (3 terms), 2 for 2 textures(8 terms) and 3 for 3 textures (15 terms)
        this._shDegree = sh ? sh.length : 0;
        const textureSize = this._getTextureSize(vertexCount);
        const textureLength = textureSize.x * textureSize.y;
        const lineCountUpdate = GaussianSplattingMesh.ProgressiveUpdateAmount ?? textureSize.y;
        const textureLengthPerUpdate = textureSize.x * lineCountUpdate;
        this._splatPositions = new Float32Array(4 * textureLength);
        const covA = new Uint16Array(textureLength * 4);
        const covB = new Uint16Array((this._useRGBACovariants ? 4 : 2) * textureLength);
        const colorArray = new Uint8Array(textureLength * 4);
        const minimum = new Vector3(Number.MAX_VALUE, Number.MAX_VALUE, Number.MAX_VALUE);
        const maximum = new Vector3(-Number.MAX_VALUE, -Number.MAX_VALUE, -Number.MAX_VALUE);
        if (GaussianSplattingMesh.ProgressiveUpdateAmount) {
            // create textures with not filled-yet array, then update directly portions of it
            this._updateTextures(covA, covB, colorArray, sh);
            this.setEnabled(true);
            const partCount = Math.ceil(textureSize.y / lineCountUpdate);
            for (let partIndex = 0; partIndex < partCount; partIndex++) {
                const updateLine = partIndex * lineCountUpdate;
                const splatIndexBase = updateLine * textureSize.x;
                for (let i = 0; i < textureLengthPerUpdate; i++) {
                    this._makeSplat(splatIndexBase + i, fBuffer, uBuffer, covA, covB, colorArray, minimum, maximum, options);
                }
                this._updateSubTextures(this._splatPositions, covA, covB, colorArray, updateLine, Math.min(lineCountUpdate, textureSize.y - updateLine));
                // Update the binfo
                this.getBoundingInfo().reConstruct(minimum, maximum, this.getWorldMatrix());
                if (isAsync) {
                    yield;
                }
            }
            // sort will be dirty here as just finished filled positions will not be sorted
            const positions = Float32Array.from(this._splatPositions);
            const vertexCount = this._vertexCount;
            if (this._worker) {
                this._worker.postMessage({ positions, vertexCount }, [positions.buffer]);
            }
            this._sortIsDirty = true;
        }
        else {
            const paddedVertexCount = (vertexCount + 15) & -16;
            for (let i = 0; i < vertexCount; i++) {
                this._makeSplat(i, fBuffer, uBuffer, covA, covB, colorArray, minimum, maximum, options);
                if (isAsync && i % GaussianSplattingMesh._SplatBatchSize === 0) {
                    yield;
                }
            }
            // pad the rest
            for (let i = vertexCount; i < paddedVertexCount; i++) {
                this._makeEmptySplat(i, covA, covB, colorArray);
            }
            // textures
            this._updateTextures(covA, covB, colorArray, sh);
            // Update the binfo
            this.getBoundingInfo().reConstruct(minimum, maximum, this.getWorldMatrix());
            this.setEnabled(true);
            this._sortIsDirty = true;
        }
        this._postToWorker(true);
    }
    /**
     * Update asynchronously the buffer
     * @param data array buffer containing center, color, orientation and scale of splats
     * @param sh optional array of uint8 array for SH data
     * @returns a promise
     */
    async updateDataAsync(data, sh) {
        return await runCoroutineAsync(this._updateData(data, true, sh), createYieldingScheduler());
    }
    /**
     * @experimental
     * Update data from GS (position, orientation, color, scaling)
     * @param data array that contain all the datas
     * @param sh optional array of uint8 array for SH data
     * @param options optional informations on how to treat data
     */
    updateData(data, sh, options = { flipY: true }) {
        runCoroutineSync(this._updateData(data, false, sh, options));
    }
    /**
     * Refreshes the bounding info, taking into account all the thin instances defined
     * @returns the current Gaussian Splatting
     */
    refreshBoundingInfo() {
        this.thinInstanceRefreshBoundingInfo(false);
        return this;
    }
    // in case size is different
    _updateSplatIndexBuffer(vertexCount) {
        const paddedVertexCount = (vertexCount + 15) & -16;
        if (!this._splatIndex || vertexCount > this._splatIndex.length) {
            this._splatIndex = new Float32Array(paddedVertexCount);
            for (let i = 0; i < paddedVertexCount; i++) {
                this._splatIndex[i] = i;
            }
            // update meshes for knowns cameras
            this._cameraViewInfos.forEach((cameraViewInfos) => {
                cameraViewInfos.mesh.thinInstanceSetBuffer("splatIndex", this._splatIndex, 16, false);
            });
        }
        this.forcedInstanceCount = paddedVertexCount >> 4;
    }
    _updateSubTextures(centers, covA, covB, colors, lineStart, lineCount, sh) {
        const updateTextureFromData = (texture, data, width, lineStart, lineCount) => {
            this.getEngine().updateTextureData(texture.getInternalTexture(), data, 0, lineStart, width, lineCount, 0, 0, false);
        };
        const textureSize = this._getTextureSize(this._vertexCount);
        const covBSItemSize = this._useRGBACovariants ? 4 : 2;
        const texelStart = lineStart * textureSize.x;
        const texelCount = lineCount * textureSize.x;
        const covAView = new Uint16Array(covA.buffer, texelStart * 4 * Uint16Array.BYTES_PER_ELEMENT, texelCount * 4);
        const covBView = new Uint16Array(covB.buffer, texelStart * covBSItemSize * Uint16Array.BYTES_PER_ELEMENT, texelCount * covBSItemSize);
        const colorsView = new Uint8Array(colors.buffer, texelStart * 4, texelCount * 4);
        const centersView = new Float32Array(centers.buffer, texelStart * 4 * Float32Array.BYTES_PER_ELEMENT, texelCount * 4);
        updateTextureFromData(this._covariancesATexture, covAView, textureSize.x, lineStart, lineCount);
        updateTextureFromData(this._covariancesBTexture, covBView, textureSize.x, lineStart, lineCount);
        updateTextureFromData(this._centersTexture, centersView, textureSize.x, lineStart, lineCount);
        updateTextureFromData(this._colorsTexture, colorsView, textureSize.x, lineStart, lineCount);
        if (sh) {
            for (let i = 0; i < sh.length; i++) {
                const componentCount = 4;
                const shView = new Uint32Array(sh[i].buffer, texelStart * componentCount * 4, texelCount * componentCount);
                updateTextureFromData(this._shTextures[i], shView, textureSize.x, lineStart, lineCount);
            }
        }
    }
    _instanciateWorker() {
        if (!this._vertexCount) {
            return;
        }
        if (this._disableDepthSort) {
            return;
        }
        this._updateSplatIndexBuffer(this._vertexCount);
        // no worker in native
        if (_native) {
            return;
        }
        // Start the worker thread
        this._worker?.terminate();
        this._worker = new Worker(URL.createObjectURL(new Blob(["(", GaussianSplattingMesh._CreateWorker.toString(), ")(self)"], {
            type: "application/javascript",
        })));
        const vertexCountPadded = (this._vertexCount + 15) & -16;
        this._depthMix = new BigInt64Array(vertexCountPadded);
        const positions = Float32Array.from(this._splatPositions);
        this._worker.postMessage({ positions }, [positions.buffer]);
        this._worker.onmessage = (e) => {
            this._depthMix = e.data.depthMix;
            const cameraId = e.data.cameraId;
            const indexMix = new Uint32Array(e.data.depthMix.buffer);
            if (this._splatIndex) {
                for (let j = 0; j < vertexCountPadded; j++) {
                    this._splatIndex[j] = indexMix[2 * j];
                }
            }
            if (this._delayedTextureUpdate) {
                const textureSize = this._getTextureSize(vertexCountPadded);
                this._updateSubTextures(this._delayedTextureUpdate.centers, this._delayedTextureUpdate.covA, this._delayedTextureUpdate.covB, this._delayedTextureUpdate.colors, 0, textureSize.y, this._delayedTextureUpdate.sh);
                this._delayedTextureUpdate = null;
            }
            // get mesh for camera and update its instance buffer
            const cameraViewInfos = this._cameraViewInfos.get(cameraId);
            if (cameraViewInfos) {
                if (cameraViewInfos.splatIndexBufferSet) {
                    cameraViewInfos.mesh.thinInstanceBufferUpdated("splatIndex");
                }
                else {
                    cameraViewInfos.mesh.thinInstanceSetBuffer("splatIndex", this._splatIndex, 16, false);
                    cameraViewInfos.splatIndexBufferSet = true;
                }
            }
            this._canPostToWorker = true;
            this._readyToDisplay = true;
            // sort is dirty when GS is visible for progressive update with a this message arriving but positions were partially filled
            // another update needs to be kicked. The kick can't happen just when the position buffer is ready because _canPostToWorker might be false.
            if (this._sortIsDirty) {
                this._postToWorker(true);
                this._sortIsDirty = false;
            }
        };
    }
    _getTextureSize(length) {
        const engine = this._scene.getEngine();
        const width = engine.getCaps().maxTextureSize;
        let height = 1;
        if (engine.version === 1 && !engine.isWebGPU) {
            while (width * height < length) {
                height *= 2;
            }
        }
        else {
            height = Math.ceil(length / width);
        }
        if (height > width) {
            Logger.Error("GaussianSplatting texture size: (" + width + ", " + height + "), maxTextureSize: " + width);
            height = width;
        }
        return new Vector2(width, height);
    }
}
GaussianSplattingMesh._RowOutputLength = 3 * 4 + 3 * 4 + 4 + 4; // Vector3 position, Vector3 scale, 1 u8 quaternion, 1 color with alpha
GaussianSplattingMesh._SH_C0 = 0.28209479177387814;
// batch size between 2 yield calls. This value is a tradeoff between updates overhead and framerate hiccups
// This step is faster the PLY conversion. So batch size can be bigger
GaussianSplattingMesh._SplatBatchSize = 327680;
// batch size between 2 yield calls during the PLY to splat conversion.
GaussianSplattingMesh._PlyConversionBatchSize = 32768;
GaussianSplattingMesh._BatchSize = 16; // 16 splats per instance
/**
 * Set the number of batch (a batch is 16384 splats) after which a display update is performed
 * A value of 0 (default) means display update will not happens before splat is ready.
 */
GaussianSplattingMesh.ProgressiveUpdateAmount = 0;
GaussianSplattingMesh._CreateWorker = function (self) {
    let positions;
    let depthMix;
    let indices;
    let floatMix;
    self.onmessage = (e) => {
        // updated on init
        if (e.data.positions) {
            positions = e.data.positions;
        }
        // udpate on view changed
        else {
            const cameraId = e.data.cameraId;
            const viewProj = e.data.view;
            const vertexCountPadded = (positions.length / 4 + 15) & -16;
            if (!positions || !viewProj) {
                // Sanity check, it shouldn't happen!
                throw new Error("positions or view is not defined!");
            }
            depthMix = e.data.depthMix;
            indices = new Uint32Array(depthMix.buffer);
            floatMix = new Float32Array(depthMix.buffer);
            // Sort
            for (let j = 0; j < vertexCountPadded; j++) {
                indices[2 * j] = j;
            }
            let depthFactor = -1;
            if (e.data.useRightHandedSystem) {
                depthFactor = 1;
            }
            for (let j = 0; j < vertexCountPadded; j++) {
                floatMix[2 * j + 1] = 10000 + (viewProj[2] * positions[4 * j + 0] + viewProj[6] * positions[4 * j + 1] + viewProj[10] * positions[4 * j + 2]) * depthFactor;
            }
            depthMix.sort();
            self.postMessage({ depthMix, cameraId }, [depthMix.buffer]);
        }
    };
};

/**
 * Represents one particle of a points cloud system.
 */
class CloudPoint {
    /**
     * Creates a Point Cloud object.
     * Don't create particles manually, use instead the PCS internal tools like _addParticle()
     * @param particleIndex (integer) is the particle index in the PCS pool. It's also the particle identifier.
     * @param group (PointsGroup) is the group the particle belongs to
     * @param groupId (integer) is the group identifier in the PCS.
     * @param idxInGroup (integer) is the index of the particle in the current point group (ex: the 10th point of addPoints(30))
     * @param pcs defines the PCS it is associated to
     */
    constructor(particleIndex, group, groupId, idxInGroup, pcs) {
        /**
         * particle global index
         */
        this.idx = 0;
        /**
         * The color of the particle
         */
        this.color = new Color4(1.0, 1.0, 1.0, 1.0);
        /**
         * The world space position of the particle.
         */
        this.position = Vector3.Zero();
        /**
         * The world space rotation of the particle. (Not use if rotationQuaternion is set)
         */
        this.rotation = Vector3.Zero();
        /**
         * The uv of the particle.
         */
        this.uv = new Vector2(0.0, 0.0);
        /**
         * The current speed of the particle.
         */
        this.velocity = Vector3.Zero();
        /**
         * The pivot point in the particle local space.
         */
        this.pivot = Vector3.Zero();
        /**
         * Must the particle be translated from its pivot point in its local space ?
         * In this case, the pivot point is set at the origin of the particle local space and the particle is translated.
         * Default : false
         */
        this.translateFromPivot = false;
        /**
         * Index of this particle in the global "positions" array (Internal use)
         * @internal
         */
        this._pos = 0;
        /**
         * @internal Index of this particle in the global "indices" array (Internal use)
         */
        this._ind = 0;
        /**
         * Group id of this particle
         */
        this.groupId = 0;
        /**
         * Index of the particle in its group id (Internal use)
         */
        this.idxInGroup = 0;
        /**
         * @internal Still set as invisible in order to skip useless computations (Internal use)
         */
        this._stillInvisible = false;
        /**
         * @internal Last computed particle rotation matrix
         */
        this._rotationMatrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        /**
         * Parent particle Id, if any.
         * Default null.
         */
        this.parentId = null;
        /**
         * @internal Internal global position in the PCS.
         */
        this._globalPosition = Vector3.Zero();
        this.idx = particleIndex;
        this._group = group;
        this.groupId = groupId;
        this.idxInGroup = idxInGroup;
        this._pcs = pcs;
    }
    /**
     * get point size
     */
    get size() {
        return this.size;
    }
    /**
     * Set point size
     */
    set size(scale) {
        this.size = scale;
    }
    /**
     * Legacy support, changed quaternion to rotationQuaternion
     */
    get quaternion() {
        return this.rotationQuaternion;
    }
    /**
     * Legacy support, changed quaternion to rotationQuaternion
     */
    set quaternion(q) {
        this.rotationQuaternion = q;
    }
    /**
     * Returns a boolean. True if the particle intersects a mesh, else false
     * The intersection is computed on the particle position and Axis Aligned Bounding Box (AABB) or Sphere
     * @param target is the object (point or mesh) what the intersection is computed against
     * @param isSphere is boolean flag when false (default) bounding box of mesh is used, when true the bounding sphere is used
     * @returns true if it intersects
     */
    intersectsMesh(target, isSphere) {
        if (!target.hasBoundingInfo) {
            return false;
        }
        if (!this._pcs.mesh) {
            throw new Error("Point Cloud System doesnt contain the Mesh");
        }
        if (isSphere) {
            return target.getBoundingInfo().boundingSphere.intersectsPoint(this.position.add(this._pcs.mesh.position));
        }
        const bbox = target.getBoundingInfo().boundingBox;
        const maxX = bbox.maximumWorld.x;
        const minX = bbox.minimumWorld.x;
        const maxY = bbox.maximumWorld.y;
        const minY = bbox.minimumWorld.y;
        const maxZ = bbox.maximumWorld.z;
        const minZ = bbox.minimumWorld.z;
        const x = this.position.x + this._pcs.mesh.position.x;
        const y = this.position.y + this._pcs.mesh.position.y;
        const z = this.position.z + this._pcs.mesh.position.z;
        return minX <= x && x <= maxX && minY <= y && y <= maxY && minZ <= z && z <= maxZ;
    }
    /**
     * get the rotation matrix of the particle
     * @internal
     */
    getRotationMatrix(m) {
        let quaternion;
        if (this.rotationQuaternion) {
            quaternion = this.rotationQuaternion;
        }
        else {
            quaternion = TmpVectors.Quaternion[0];
            const rotation = this.rotation;
            Quaternion.RotationYawPitchRollToRef(rotation.y, rotation.x, rotation.z, quaternion);
        }
        quaternion.toRotationMatrix(m);
    }
}
/**
 * Represents a group of points in a points cloud system
 *  * PCS internal tool, don't use it manually.
 */
class PointsGroup {
    /**
     * Get or set the groupId
     * @deprecated Please use groupId instead
     */
    // eslint-disable-next-line @typescript-eslint/naming-convention
    get groupID() {
        return this.groupId;
    }
    // eslint-disable-next-line @typescript-eslint/naming-convention
    set groupID(groupID) {
        this.groupId = groupID;
    }
    /**
     * Creates a points group object. This is an internal reference to produce particles for the PCS.
     * PCS internal tool, don't use it manually.
     * @internal
     */
    constructor(id, posFunction) {
        this.groupId = id;
        this._positionFunction = posFunction;
    }
}

/** Defines the 4 color options */
var PointColor;
(function (PointColor) {
    /** color value */
    PointColor[PointColor["Color"] = 2] = "Color";
    /** uv value */
    PointColor[PointColor["UV"] = 1] = "UV";
    /** random value */
    PointColor[PointColor["Random"] = 0] = "Random";
    /** stated value */
    PointColor[PointColor["Stated"] = 3] = "Stated";
})(PointColor || (PointColor = {}));
/**
 * The PointCloudSystem (PCS) is a single updatable mesh. The points corresponding to the vertices of this big mesh.
 * As it is just a mesh, the PointCloudSystem has all the same properties as any other BJS mesh : not more, not less. It can be scaled, rotated, translated, enlighted, textured, moved, etc.

 * The PointCloudSystem is also a particle system, with each point being a particle. It provides some methods to manage the particles.
 * However it is behavior agnostic. This means it has no emitter, no particle physics, no particle recycler. You have to implement your own behavior.
 *
 * Full documentation here : TO BE ENTERED
 */
class PointsCloudSystem {
    /**
     * Gets the particle positions computed by the Point Cloud System
     */
    get positions() {
        return this._positions32;
    }
    /**
     * Gets the particle colors computed by the Point Cloud System
     */
    get colors() {
        return this._colors32;
    }
    /**
     * Gets the particle uvs computed by the Point Cloud System
     */
    get uvs() {
        return this._uvs32;
    }
    /**
     * Creates a PCS (Points Cloud System) object
     * @param name (String) is the PCS name, this will be the underlying mesh name
     * @param pointSize (number) is the size for each point. Has no effect on a WebGPU engine.
     * @param scene (Scene) is the scene in which the PCS is added
     * @param options defines the options of the PCS e.g.
     * * updatable (optional boolean, default true) : if the PCS must be updatable or immutable
     */
    constructor(name, pointSize, scene, options) {
        /**
         *  The PCS array of cloud point objects. Just access each particle as with any classic array.
         *  Example : var p = SPS.particles[i];
         */
        this.particles = new Array();
        /**
         * The PCS total number of particles. Read only. Use PCS.counter instead if you need to set your own value.
         */
        this.nbParticles = 0;
        /**
         * This a counter for your own usage. It's not set by any SPS functions.
         */
        this.counter = 0;
        /**
         * This empty object is intended to store some PCS specific or temporary values in order to lower the Garbage Collector activity.
         * Please read :
         */
        this.vars = {};
        this._promises = [];
        this._positions = new Array();
        this._indices = new Array();
        this._normals = new Array();
        this._colors = new Array();
        this._uvs = new Array();
        this._updatable = true;
        this._isVisibilityBoxLocked = false;
        this._alwaysVisible = false;
        this._groups = new Array(); //start indices for each group of particles
        this._groupCounter = 0;
        this._computeParticleColor = true;
        this._computeParticleTexture = true;
        this._computeParticleRotation = true;
        this._computeBoundingBox = false;
        this._isReady = false;
        this.name = name;
        this._size = pointSize;
        this._scene = scene || EngineStore.LastCreatedScene;
        if (options && options.updatable !== undefined) {
            this._updatable = options.updatable;
        }
        else {
            this._updatable = true;
        }
    }
    /**
     * Builds the PCS underlying mesh. Returns a standard Mesh.
     * If no points were added to the PCS, the returned mesh is just a single point.
     * @param material The material to use to render the mesh. If not provided, will create a default one
     * @returns a promise for the created mesh
     */
    async buildMeshAsync(material) {
        await Promise.all(this._promises);
        this._isReady = true;
        return await this._buildMeshAsync(material);
    }
    async _buildMeshAsync(material) {
        if (this.nbParticles === 0) {
            this.addPoints(1);
        }
        this._positions32 = new Float32Array(this._positions);
        this._uvs32 = new Float32Array(this._uvs);
        this._colors32 = new Float32Array(this._colors);
        const vertexData = new VertexData();
        vertexData.set(this._positions32, VertexBuffer.PositionKind);
        if (this._uvs32.length > 0) {
            vertexData.set(this._uvs32, VertexBuffer.UVKind);
        }
        let ec = 0; //emissive color value 0 for UVs, 1 for color
        if (this._colors32.length > 0) {
            ec = 1;
            vertexData.set(this._colors32, VertexBuffer.ColorKind);
        }
        const mesh = new Mesh(this.name, this._scene);
        vertexData.applyToMesh(mesh, this._updatable);
        this.mesh = mesh;
        // free memory
        this._positions = null;
        this._uvs = null;
        this._colors = null;
        if (!this._updatable) {
            this.particles.length = 0;
        }
        let mat = material;
        if (!mat) {
            mat = new StandardMaterial("point cloud material", this._scene);
            mat.emissiveColor = new Color3(ec, ec, ec);
            mat.disableLighting = true;
            mat.pointsCloud = true;
            mat.pointSize = this._size;
        }
        mesh.material = mat;
        return mesh;
    }
    // adds a new particle object in the particles array
    _addParticle(idx, group, groupId, idxInGroup) {
        const cp = new CloudPoint(idx, group, groupId, idxInGroup, this);
        this.particles.push(cp);
        return cp;
    }
    _randomUnitVector(particle) {
        particle.position = new Vector3(Math.random(), Math.random(), Math.random());
        particle.color = new Color4(1, 1, 1, 1);
    }
    _getColorIndicesForCoord(pointsGroup, x, y, width) {
        const imageData = pointsGroup._groupImageData;
        const color = y * (width * 4) + x * 4;
        const colorIndices = [color, color + 1, color + 2, color + 3];
        const redIndex = colorIndices[0];
        const greenIndex = colorIndices[1];
        const blueIndex = colorIndices[2];
        const alphaIndex = colorIndices[3];
        const redForCoord = imageData[redIndex];
        const greenForCoord = imageData[greenIndex];
        const blueForCoord = imageData[blueIndex];
        const alphaForCoord = imageData[alphaIndex];
        return new Color4(redForCoord / 255, greenForCoord / 255, blueForCoord / 255, alphaForCoord);
    }
    _setPointsColorOrUV(mesh, pointsGroup, isVolume, colorFromTexture, hasTexture, color, range, uvSetIndex) {
        uvSetIndex = uvSetIndex ?? 0;
        if (isVolume) {
            mesh.updateFacetData();
        }
        const boundInfo = mesh.getBoundingInfo();
        const diameter = 2 * boundInfo.boundingSphere.radius;
        let meshPos = mesh.getVerticesData(VertexBuffer.PositionKind);
        const meshInd = mesh.getIndices();
        const meshUV = mesh.getVerticesData(VertexBuffer.UVKind + (uvSetIndex ? uvSetIndex + 1 : ""));
        const meshCol = mesh.getVerticesData(VertexBuffer.ColorKind);
        const place = Vector3.Zero();
        mesh.computeWorldMatrix();
        const meshMatrix = mesh.getWorldMatrix();
        if (!meshMatrix.isIdentity()) {
            meshPos = meshPos.slice(0);
            for (let p = 0; p < meshPos.length / 3; p++) {
                Vector3.TransformCoordinatesFromFloatsToRef(meshPos[3 * p], meshPos[3 * p + 1], meshPos[3 * p + 2], meshMatrix, place);
                meshPos[3 * p] = place.x;
                meshPos[3 * p + 1] = place.y;
                meshPos[3 * p + 2] = place.z;
            }
        }
        let idxPoints = 0;
        let id0 = 0;
        let id1 = 0;
        let id2 = 0;
        let v0X = 0;
        let v0Y = 0;
        let v0Z = 0;
        let v1X = 0;
        let v1Y = 0;
        let v1Z = 0;
        let v2X = 0;
        let v2Y = 0;
        let v2Z = 0;
        const vertex0 = Vector3.Zero();
        const vertex1 = Vector3.Zero();
        const vertex2 = Vector3.Zero();
        const vec0 = Vector3.Zero();
        const vec1 = Vector3.Zero();
        let uv0X = 0;
        let uv0Y = 0;
        let uv1X = 0;
        let uv1Y = 0;
        let uv2X = 0;
        let uv2Y = 0;
        const uv0 = Vector2.Zero();
        const uv1 = Vector2.Zero();
        const uv2 = Vector2.Zero();
        const uvec0 = Vector2.Zero();
        const uvec1 = Vector2.Zero();
        let col0X = 0;
        let col0Y = 0;
        let col0Z = 0;
        let col0A = 0;
        let col1X = 0;
        let col1Y = 0;
        let col1Z = 0;
        let col1A = 0;
        let col2X = 0;
        let col2Y = 0;
        let col2Z = 0;
        let col2A = 0;
        const col0 = Vector4.Zero();
        const col1 = Vector4.Zero();
        const col2 = Vector4.Zero();
        const colvec0 = Vector4.Zero();
        const colvec1 = Vector4.Zero();
        let lamda = 0;
        let mu = 0;
        range = range ? range : 0;
        let facetPoint;
        let uvPoint;
        let colPoint = new Vector4(0, 0, 0, 0);
        let norm = Vector3.Zero();
        let tang = Vector3.Zero();
        let biNorm = Vector3.Zero();
        let angle = 0;
        let facetPlaneVec = Vector3.Zero();
        let gap = 0;
        let distance = 0;
        const ray = new Ray(Vector3.Zero(), new Vector3(1, 0, 0));
        let pickInfo;
        let direction = Vector3.Zero();
        for (let index = 0; index < meshInd.length / 3; index++) {
            id0 = meshInd[3 * index];
            id1 = meshInd[3 * index + 1];
            id2 = meshInd[3 * index + 2];
            v0X = meshPos[3 * id0];
            v0Y = meshPos[3 * id0 + 1];
            v0Z = meshPos[3 * id0 + 2];
            v1X = meshPos[3 * id1];
            v1Y = meshPos[3 * id1 + 1];
            v1Z = meshPos[3 * id1 + 2];
            v2X = meshPos[3 * id2];
            v2Y = meshPos[3 * id2 + 1];
            v2Z = meshPos[3 * id2 + 2];
            vertex0.set(v0X, v0Y, v0Z);
            vertex1.set(v1X, v1Y, v1Z);
            vertex2.set(v2X, v2Y, v2Z);
            vertex1.subtractToRef(vertex0, vec0);
            vertex2.subtractToRef(vertex1, vec1);
            if (meshUV) {
                uv0X = meshUV[2 * id0];
                uv0Y = meshUV[2 * id0 + 1];
                uv1X = meshUV[2 * id1];
                uv1Y = meshUV[2 * id1 + 1];
                uv2X = meshUV[2 * id2];
                uv2Y = meshUV[2 * id2 + 1];
                uv0.set(uv0X, uv0Y);
                uv1.set(uv1X, uv1Y);
                uv2.set(uv2X, uv2Y);
                uv1.subtractToRef(uv0, uvec0);
                uv2.subtractToRef(uv1, uvec1);
            }
            if (meshCol && colorFromTexture) {
                col0X = meshCol[4 * id0];
                col0Y = meshCol[4 * id0 + 1];
                col0Z = meshCol[4 * id0 + 2];
                col0A = meshCol[4 * id0 + 3];
                col1X = meshCol[4 * id1];
                col1Y = meshCol[4 * id1 + 1];
                col1Z = meshCol[4 * id1 + 2];
                col1A = meshCol[4 * id1 + 3];
                col2X = meshCol[4 * id2];
                col2Y = meshCol[4 * id2 + 1];
                col2Z = meshCol[4 * id2 + 2];
                col2A = meshCol[4 * id2 + 3];
                col0.set(col0X, col0Y, col0Z, col0A);
                col1.set(col1X, col1Y, col1Z, col1A);
                col2.set(col2X, col2Y, col2Z, col2A);
                col1.subtractToRef(col0, colvec0);
                col2.subtractToRef(col1, colvec1);
            }
            let width;
            let height;
            let deltaS;
            let deltaV;
            let h;
            let s;
            let v;
            let hsvCol;
            const statedColor = new Color3(0, 0, 0);
            const colPoint3 = new Color3(0, 0, 0);
            let pointColors;
            let particle;
            for (let i = 0; i < pointsGroup._groupDensity[index]; i++) {
                idxPoints = this.particles.length;
                this._addParticle(idxPoints, pointsGroup, this._groupCounter, index + i);
                particle = this.particles[idxPoints];
                //form a point inside the facet v0, v1, v2;
                lamda = Math.sqrt(RandomRange(0, 1));
                mu = RandomRange(0, 1);
                facetPoint = vertex0.add(vec0.scale(lamda)).add(vec1.scale(lamda * mu));
                if (isVolume) {
                    norm = mesh.getFacetNormal(index).normalize().scale(-1);
                    tang = vec0.clone().normalize();
                    biNorm = Vector3.Cross(norm, tang);
                    angle = RandomRange(0, 2 * Math.PI);
                    facetPlaneVec = tang.scale(Math.cos(angle)).add(biNorm.scale(Math.sin(angle)));
                    angle = RandomRange(0.1, Math.PI / 2);
                    direction = facetPlaneVec.scale(Math.cos(angle)).add(norm.scale(Math.sin(angle)));
                    ray.origin = facetPoint.add(direction.scale(0.00001));
                    ray.direction = direction;
                    ray.length = diameter;
                    pickInfo = ray.intersectsMesh(mesh);
                    if (pickInfo.hit) {
                        distance = pickInfo.pickedPoint.subtract(facetPoint).length();
                        gap = RandomRange(0, 1) * distance;
                        facetPoint.addInPlace(direction.scale(gap));
                    }
                }
                particle.position = facetPoint.clone();
                this._positions.push(particle.position.x, particle.position.y, particle.position.z);
                if (colorFromTexture !== undefined) {
                    if (meshUV) {
                        uvPoint = uv0.add(uvec0.scale(lamda)).add(uvec1.scale(lamda * mu));
                        if (colorFromTexture) {
                            //Set particle color to texture color
                            if (hasTexture && pointsGroup._groupImageData !== null) {
                                width = pointsGroup._groupImgWidth;
                                height = pointsGroup._groupImgHeight;
                                pointColors = this._getColorIndicesForCoord(pointsGroup, Math.round(uvPoint.x * width), Math.round(uvPoint.y * height), width);
                                particle.color = pointColors;
                                this._colors.push(pointColors.r, pointColors.g, pointColors.b, pointColors.a);
                            }
                            else {
                                if (meshCol) {
                                    //failure in texture and colors available
                                    colPoint = col0.add(colvec0.scale(lamda)).add(colvec1.scale(lamda * mu));
                                    particle.color = new Color4(colPoint.x, colPoint.y, colPoint.z, colPoint.w);
                                    this._colors.push(colPoint.x, colPoint.y, colPoint.z, colPoint.w);
                                }
                                else {
                                    colPoint = col0.set(Math.random(), Math.random(), Math.random(), 1);
                                    particle.color = new Color4(colPoint.x, colPoint.y, colPoint.z, colPoint.w);
                                    this._colors.push(colPoint.x, colPoint.y, colPoint.z, colPoint.w);
                                }
                            }
                        }
                        else {
                            //Set particle uv based on a mesh uv
                            particle.uv = uvPoint.clone();
                            this._uvs.push(particle.uv.x, particle.uv.y);
                        }
                    }
                }
                else {
                    if (color) {
                        statedColor.set(color.r, color.g, color.b);
                        deltaS = RandomRange(-range, range);
                        deltaV = RandomRange(-range, range);
                        hsvCol = statedColor.toHSV();
                        h = hsvCol.r;
                        s = hsvCol.g + deltaS;
                        v = hsvCol.b + deltaV;
                        if (s < 0) {
                            s = 0;
                        }
                        if (s > 1) {
                            s = 1;
                        }
                        if (v < 0) {
                            v = 0;
                        }
                        if (v > 1) {
                            v = 1;
                        }
                        Color3.HSVtoRGBToRef(h, s, v, colPoint3);
                        colPoint.set(colPoint3.r, colPoint3.g, colPoint3.b, 1);
                    }
                    else {
                        colPoint = col0.set(Math.random(), Math.random(), Math.random(), 1);
                    }
                    particle.color = new Color4(colPoint.x, colPoint.y, colPoint.z, colPoint.w);
                    this._colors.push(colPoint.x, colPoint.y, colPoint.z, colPoint.w);
                }
            }
        }
    }
    // stores mesh texture in dynamic texture for color pixel retrieval
    // when pointColor type is color for surface points
    _colorFromTexture(mesh, pointsGroup, isVolume) {
        if (mesh.material === null) {
            Logger.Warn(mesh.name + "has no material.");
            pointsGroup._groupImageData = null;
            this._setPointsColorOrUV(mesh, pointsGroup, isVolume, true, false);
            return;
        }
        const mat = mesh.material;
        const textureList = mat.getActiveTextures();
        if (textureList.length === 0) {
            Logger.Warn(mesh.name + "has no usable texture.");
            pointsGroup._groupImageData = null;
            this._setPointsColorOrUV(mesh, pointsGroup, isVolume, true, false);
            return;
        }
        const clone = mesh.clone();
        clone.setEnabled(false);
        this._promises.push(new Promise((resolve) => {
            BaseTexture.WhenAllReady(textureList, () => {
                let n = pointsGroup._textureNb;
                if (n < 0) {
                    n = 0;
                }
                if (n > textureList.length - 1) {
                    n = textureList.length - 1;
                }
                const finalize = () => {
                    pointsGroup._groupImgWidth = textureList[n].getSize().width;
                    pointsGroup._groupImgHeight = textureList[n].getSize().height;
                    this._setPointsColorOrUV(clone, pointsGroup, isVolume, true, true, undefined, undefined, textureList[n].coordinatesIndex);
                    clone.dispose();
                    resolve();
                };
                pointsGroup._groupImageData = null;
                const dataPromise = textureList[n].readPixels();
                if (!dataPromise) {
                    finalize();
                }
                else {
                    // eslint-disable-next-line @typescript-eslint/no-floating-promises, github/no-then
                    dataPromise.then((data) => {
                        pointsGroup._groupImageData = data;
                        finalize();
                    });
                }
            });
        }));
    }
    // calculates the point density per facet of a mesh for surface points
    _calculateDensity(nbPoints, positions, indices) {
        let id0;
        let id1;
        let id2;
        let v0X;
        let v0Y;
        let v0Z;
        let v1X;
        let v1Y;
        let v1Z;
        let v2X;
        let v2Y;
        let v2Z;
        const vertex0 = Vector3.Zero();
        const vertex1 = Vector3.Zero();
        const vertex2 = Vector3.Zero();
        const vec0 = Vector3.Zero();
        const vec1 = Vector3.Zero();
        const normal = Vector3.Zero();
        let area;
        const cumulativeAreas = [];
        let surfaceArea = 0;
        const nbFacets = indices.length / 3;
        //surface area
        for (let index = 0; index < nbFacets; index++) {
            id0 = indices[3 * index];
            id1 = indices[3 * index + 1];
            id2 = indices[3 * index + 2];
            v0X = positions[3 * id0];
            v0Y = positions[3 * id0 + 1];
            v0Z = positions[3 * id0 + 2];
            v1X = positions[3 * id1];
            v1Y = positions[3 * id1 + 1];
            v1Z = positions[3 * id1 + 2];
            v2X = positions[3 * id2];
            v2Y = positions[3 * id2 + 1];
            v2Z = positions[3 * id2 + 2];
            vertex0.set(v0X, v0Y, v0Z);
            vertex1.set(v1X, v1Y, v1Z);
            vertex2.set(v2X, v2Y, v2Z);
            vertex1.subtractToRef(vertex0, vec0);
            vertex2.subtractToRef(vertex1, vec1);
            Vector3.CrossToRef(vec0, vec1, normal);
            area = 0.5 * normal.length();
            surfaceArea += area;
            cumulativeAreas[index] = surfaceArea;
        }
        const density = new Array(nbFacets);
        let remainingPoints = nbPoints;
        for (let index = nbFacets - 1; index > 0; index--) {
            const cumulativeArea = cumulativeAreas[index];
            if (cumulativeArea === 0) {
                // avoiding division by 0 upon degenerate triangles
                density[index] = 0;
            }
            else {
                const area = cumulativeArea - cumulativeAreas[index - 1];
                const facetPointsWithFraction = (area / cumulativeArea) * remainingPoints;
                const floored = Math.floor(facetPointsWithFraction);
                const fraction = facetPointsWithFraction - floored;
                const extraPoint = Number(Math.random() < fraction);
                const facetPoints = floored + extraPoint;
                density[index] = facetPoints;
                remainingPoints -= facetPoints;
            }
        }
        density[0] = remainingPoints;
        return density;
    }
    /**
     * Adds points to the PCS in random positions within a unit sphere
     * @param nb (positive integer) the number of particles to be created from this model
     * @param pointFunction is an optional javascript function to be called for each particle on PCS creation
     * @returns the number of groups in the system
     */
    addPoints(nb, pointFunction = this._randomUnitVector) {
        const pointsGroup = new PointsGroup(this._groupCounter, pointFunction);
        let cp;
        // particles
        let idx = this.nbParticles;
        for (let i = 0; i < nb; i++) {
            cp = this._addParticle(idx, pointsGroup, this._groupCounter, i);
            if (pointsGroup && pointsGroup._positionFunction) {
                pointsGroup._positionFunction(cp, idx, i);
            }
            this._positions.push(cp.position.x, cp.position.y, cp.position.z);
            if (cp.color) {
                this._colors.push(cp.color.r, cp.color.g, cp.color.b, cp.color.a);
            }
            if (cp.uv) {
                this._uvs.push(cp.uv.x, cp.uv.y);
            }
            idx++;
        }
        this.nbParticles += nb;
        this._groupCounter++;
        return this._groupCounter;
    }
    /**
     * Adds points to the PCS from the surface of the model shape
     * @param mesh is any Mesh object that will be used as a surface model for the points
     * @param nb (positive integer) the number of particles to be created from this model
     * @param colorWith determines whether a point is colored using color (default), uv, random, stated or none (invisible)
     * @param color (color4) to be used when colorWith is stated or color (number) when used to specify texture position
     * @param range (number from 0 to 1) to determine the variation in shape and tone for a stated color
     * @returns the number of groups in the system
     */
    addSurfacePoints(mesh, nb, colorWith, color, range) {
        let colored = colorWith ? colorWith : 0 /* PointColor.Random */;
        if (isNaN(colored) || colored < 0 || colored > 3) {
            colored = 0 /* PointColor.Random */;
        }
        const meshPos = mesh.getVerticesData(VertexBuffer.PositionKind);
        const meshInd = mesh.getIndices();
        this._groups.push(this._groupCounter);
        const pointsGroup = new PointsGroup(this._groupCounter, null);
        pointsGroup._groupDensity = this._calculateDensity(nb, meshPos, meshInd);
        if (colored === 2 /* PointColor.Color */) {
            pointsGroup._textureNb = color ? color : 0;
        }
        else {
            color = color ? color : new Color4(1, 1, 1, 1);
        }
        switch (colored) {
            case 2 /* PointColor.Color */:
                this._colorFromTexture(mesh, pointsGroup, false);
                break;
            case 1 /* PointColor.UV */:
                this._setPointsColorOrUV(mesh, pointsGroup, false, false, false);
                break;
            case 0 /* PointColor.Random */:
                this._setPointsColorOrUV(mesh, pointsGroup, false);
                break;
            case 3 /* PointColor.Stated */:
                this._setPointsColorOrUV(mesh, pointsGroup, false, undefined, undefined, color, range);
                break;
        }
        this.nbParticles += nb;
        this._groupCounter++;
        return this._groupCounter - 1;
    }
    /**
     * Adds points to the PCS inside the model shape
     * @param mesh is any Mesh object that will be used as a surface model for the points
     * @param nb (positive integer) the number of particles to be created from this model
     * @param colorWith determines whether a point is colored using color (default), uv, random, stated or none (invisible)
     * @param color (color4) to be used when colorWith is stated or color (number) when used to specify texture position
     * @param range (number from 0 to 1) to determine the variation in shape and tone for a stated color
     * @returns the number of groups in the system
     */
    addVolumePoints(mesh, nb, colorWith, color, range) {
        let colored = colorWith ? colorWith : 0 /* PointColor.Random */;
        if (isNaN(colored) || colored < 0 || colored > 3) {
            colored = 0 /* PointColor.Random */;
        }
        const meshPos = mesh.getVerticesData(VertexBuffer.PositionKind);
        const meshInd = mesh.getIndices();
        this._groups.push(this._groupCounter);
        const pointsGroup = new PointsGroup(this._groupCounter, null);
        pointsGroup._groupDensity = this._calculateDensity(nb, meshPos, meshInd);
        if (colored === 2 /* PointColor.Color */) {
            pointsGroup._textureNb = color ? color : 0;
        }
        else {
            color = color ? color : new Color4(1, 1, 1, 1);
        }
        switch (colored) {
            case 2 /* PointColor.Color */:
                this._colorFromTexture(mesh, pointsGroup, true);
                break;
            case 1 /* PointColor.UV */:
                this._setPointsColorOrUV(mesh, pointsGroup, true, false, false);
                break;
            case 0 /* PointColor.Random */:
                this._setPointsColorOrUV(mesh, pointsGroup, true);
                break;
            case 3 /* PointColor.Stated */:
                this._setPointsColorOrUV(mesh, pointsGroup, true, undefined, undefined, color, range);
                break;
        }
        this.nbParticles += nb;
        this._groupCounter++;
        return this._groupCounter - 1;
    }
    /**
     *  Sets all the particles : this method actually really updates the mesh according to the particle positions, rotations, colors, textures, etc.
     *  This method calls `updateParticle()` for each particle of the SPS.
     *  For an animated SPS, it is usually called within the render loop.
     * @param start The particle index in the particle array where to start to compute the particle property values _(default 0)_
     * @param end The particle index in the particle array where to stop to compute the particle property values _(default nbParticle - 1)_
     * @param update If the mesh must be finally updated on this call after all the particle computations _(default true)_
     * @returns the PCS.
     */
    setParticles(start = 0, end = this.nbParticles - 1, update = true) {
        if (!this._updatable || !this._isReady) {
            return this;
        }
        // custom beforeUpdate
        this.beforeUpdateParticles(start, end, update);
        const rotMatrix = TmpVectors.Matrix[0];
        const mesh = this.mesh;
        const colors32 = this._colors32;
        const positions32 = this._positions32;
        const uvs32 = this._uvs32;
        const tempVectors = TmpVectors.Vector3;
        const camAxisX = tempVectors[5].copyFromFloats(1.0, 0.0, 0.0);
        const camAxisY = tempVectors[6].copyFromFloats(0.0, 1.0, 0.0);
        const camAxisZ = tempVectors[7].copyFromFloats(0.0, 0.0, 1.0);
        const minimum = tempVectors[8].setAll(Number.MAX_VALUE);
        const maximum = tempVectors[9].setAll(-Number.MAX_VALUE);
        Matrix.IdentityToRef(rotMatrix);
        let idx = 0; // current index of the particle
        if (this.mesh?.isFacetDataEnabled) {
            this._computeBoundingBox = true;
        }
        end = end >= this.nbParticles ? this.nbParticles - 1 : end;
        if (this._computeBoundingBox) {
            if (start != 0 || end != this.nbParticles - 1) {
                // only some particles are updated, then use the current existing BBox basis. Note : it can only increase.
                const boundingInfo = this.mesh?.getBoundingInfo();
                if (boundingInfo) {
                    minimum.copyFrom(boundingInfo.minimum);
                    maximum.copyFrom(boundingInfo.maximum);
                }
            }
        }
        idx = 0; // particle index
        let pindex = 0; //index in positions array
        let cindex = 0; //index in color array
        let uindex = 0; //index in uv array
        // particle loop
        for (let p = start; p <= end; p++) {
            const particle = this.particles[p];
            idx = particle.idx;
            pindex = 3 * idx;
            cindex = 4 * idx;
            uindex = 2 * idx;
            // call to custom user function to update the particle properties
            this.updateParticle(particle);
            const particleRotationMatrix = particle._rotationMatrix;
            const particlePosition = particle.position;
            const particleGlobalPosition = particle._globalPosition;
            if (this._computeParticleRotation) {
                particle.getRotationMatrix(rotMatrix);
            }
            const particleHasParent = particle.parentId !== null;
            if (particleHasParent) {
                const parent = this.particles[particle.parentId];
                const parentRotationMatrix = parent._rotationMatrix;
                const parentGlobalPosition = parent._globalPosition;
                const rotatedY = particlePosition.x * parentRotationMatrix[1] + particlePosition.y * parentRotationMatrix[4] + particlePosition.z * parentRotationMatrix[7];
                const rotatedX = particlePosition.x * parentRotationMatrix[0] + particlePosition.y * parentRotationMatrix[3] + particlePosition.z * parentRotationMatrix[6];
                const rotatedZ = particlePosition.x * parentRotationMatrix[2] + particlePosition.y * parentRotationMatrix[5] + particlePosition.z * parentRotationMatrix[8];
                particleGlobalPosition.x = parentGlobalPosition.x + rotatedX;
                particleGlobalPosition.y = parentGlobalPosition.y + rotatedY;
                particleGlobalPosition.z = parentGlobalPosition.z + rotatedZ;
                if (this._computeParticleRotation) {
                    const rotMatrixValues = rotMatrix.m;
                    particleRotationMatrix[0] =
                        rotMatrixValues[0] * parentRotationMatrix[0] + rotMatrixValues[1] * parentRotationMatrix[3] + rotMatrixValues[2] * parentRotationMatrix[6];
                    particleRotationMatrix[1] =
                        rotMatrixValues[0] * parentRotationMatrix[1] + rotMatrixValues[1] * parentRotationMatrix[4] + rotMatrixValues[2] * parentRotationMatrix[7];
                    particleRotationMatrix[2] =
                        rotMatrixValues[0] * parentRotationMatrix[2] + rotMatrixValues[1] * parentRotationMatrix[5] + rotMatrixValues[2] * parentRotationMatrix[8];
                    particleRotationMatrix[3] =
                        rotMatrixValues[4] * parentRotationMatrix[0] + rotMatrixValues[5] * parentRotationMatrix[3] + rotMatrixValues[6] * parentRotationMatrix[6];
                    particleRotationMatrix[4] =
                        rotMatrixValues[4] * parentRotationMatrix[1] + rotMatrixValues[5] * parentRotationMatrix[4] + rotMatrixValues[6] * parentRotationMatrix[7];
                    particleRotationMatrix[5] =
                        rotMatrixValues[4] * parentRotationMatrix[2] + rotMatrixValues[5] * parentRotationMatrix[5] + rotMatrixValues[6] * parentRotationMatrix[8];
                    particleRotationMatrix[6] =
                        rotMatrixValues[8] * parentRotationMatrix[0] + rotMatrixValues[9] * parentRotationMatrix[3] + rotMatrixValues[10] * parentRotationMatrix[6];
                    particleRotationMatrix[7] =
                        rotMatrixValues[8] * parentRotationMatrix[1] + rotMatrixValues[9] * parentRotationMatrix[4] + rotMatrixValues[10] * parentRotationMatrix[7];
                    particleRotationMatrix[8] =
                        rotMatrixValues[8] * parentRotationMatrix[2] + rotMatrixValues[9] * parentRotationMatrix[5] + rotMatrixValues[10] * parentRotationMatrix[8];
                }
            }
            else {
                particleGlobalPosition.x = 0;
                particleGlobalPosition.y = 0;
                particleGlobalPosition.z = 0;
                if (this._computeParticleRotation) {
                    const rotMatrixValues = rotMatrix.m;
                    particleRotationMatrix[0] = rotMatrixValues[0];
                    particleRotationMatrix[1] = rotMatrixValues[1];
                    particleRotationMatrix[2] = rotMatrixValues[2];
                    particleRotationMatrix[3] = rotMatrixValues[4];
                    particleRotationMatrix[4] = rotMatrixValues[5];
                    particleRotationMatrix[5] = rotMatrixValues[6];
                    particleRotationMatrix[6] = rotMatrixValues[8];
                    particleRotationMatrix[7] = rotMatrixValues[9];
                    particleRotationMatrix[8] = rotMatrixValues[10];
                }
            }
            const pivotBackTranslation = tempVectors[11];
            if (particle.translateFromPivot) {
                pivotBackTranslation.setAll(0.0);
            }
            else {
                pivotBackTranslation.copyFrom(particle.pivot);
            }
            // positions
            const tmpVertex = tempVectors[0];
            tmpVertex.copyFrom(particle.position);
            const vertexX = tmpVertex.x - particle.pivot.x;
            const vertexY = tmpVertex.y - particle.pivot.y;
            const vertexZ = tmpVertex.z - particle.pivot.z;
            let rotatedX = vertexX * particleRotationMatrix[0] + vertexY * particleRotationMatrix[3] + vertexZ * particleRotationMatrix[6];
            let rotatedY = vertexX * particleRotationMatrix[1] + vertexY * particleRotationMatrix[4] + vertexZ * particleRotationMatrix[7];
            let rotatedZ = vertexX * particleRotationMatrix[2] + vertexY * particleRotationMatrix[5] + vertexZ * particleRotationMatrix[8];
            rotatedX += pivotBackTranslation.x;
            rotatedY += pivotBackTranslation.y;
            rotatedZ += pivotBackTranslation.z;
            const px = (positions32[pindex] = particleGlobalPosition.x + camAxisX.x * rotatedX + camAxisY.x * rotatedY + camAxisZ.x * rotatedZ);
            const py = (positions32[pindex + 1] = particleGlobalPosition.y + camAxisX.y * rotatedX + camAxisY.y * rotatedY + camAxisZ.y * rotatedZ);
            const pz = (positions32[pindex + 2] = particleGlobalPosition.z + camAxisX.z * rotatedX + camAxisY.z * rotatedY + camAxisZ.z * rotatedZ);
            if (this._computeBoundingBox) {
                minimum.minimizeInPlaceFromFloats(px, py, pz);
                maximum.maximizeInPlaceFromFloats(px, py, pz);
            }
            if (this._computeParticleColor && particle.color) {
                const color = particle.color;
                const colors32 = this._colors32;
                colors32[cindex] = color.r;
                colors32[cindex + 1] = color.g;
                colors32[cindex + 2] = color.b;
                colors32[cindex + 3] = color.a;
            }
            if (this._computeParticleTexture && particle.uv) {
                const uv = particle.uv;
                const uvs32 = this._uvs32;
                uvs32[uindex] = uv.x;
                uvs32[uindex + 1] = uv.y;
            }
        }
        // if the VBO must be updated
        if (mesh) {
            if (update) {
                if (this._computeParticleColor) {
                    mesh.updateVerticesData(VertexBuffer.ColorKind, colors32, false, false);
                }
                if (this._computeParticleTexture) {
                    mesh.updateVerticesData(VertexBuffer.UVKind, uvs32, false, false);
                }
                mesh.updateVerticesData(VertexBuffer.PositionKind, positions32, false, false);
            }
            if (this._computeBoundingBox) {
                if (mesh.hasBoundingInfo) {
                    mesh.getBoundingInfo().reConstruct(minimum, maximum, mesh._worldMatrix);
                }
                else {
                    mesh.buildBoundingInfo(minimum, maximum, mesh._worldMatrix);
                }
            }
        }
        this.afterUpdateParticles(start, end, update);
        return this;
    }
    /**
     * Disposes the PCS.
     */
    dispose() {
        this.mesh?.dispose();
        this.vars = null;
        // drop references to internal big arrays for the GC
        this._positions = null;
        this._indices = null;
        this._normals = null;
        this._uvs = null;
        this._colors = null;
        this._indices32 = null;
        this._positions32 = null;
        this._uvs32 = null;
        this._colors32 = null;
    }
    /**
     * Visibility helper : Recomputes the visible size according to the mesh bounding box
     * doc :
     * @returns the PCS.
     */
    refreshVisibleSize() {
        if (!this._isVisibilityBoxLocked) {
            this.mesh?.refreshBoundingInfo();
        }
        return this;
    }
    /**
     * Visibility helper : Sets the size of a visibility box, this sets the underlying mesh bounding box.
     * @param size the size (float) of the visibility box
     * note : this doesn't lock the PCS mesh bounding box.
     * doc :
     */
    setVisibilityBox(size) {
        if (!this.mesh) {
            return;
        }
        const vis = size / 2;
        this.mesh.buildBoundingInfo(new Vector3(-vis, -vis, -vis), new Vector3(vis, vis, vis));
    }
    /**
     * Gets whether the PCS is always visible or not
     * doc :
     */
    get isAlwaysVisible() {
        return this._alwaysVisible;
    }
    /**
     * Sets the PCS as always visible or not
     * doc :
     */
    set isAlwaysVisible(val) {
        if (!this.mesh) {
            return;
        }
        this._alwaysVisible = val;
        this.mesh.alwaysSelectAsActiveMesh = val;
    }
    /**
     * Tells to `setParticles()` to compute the particle rotations or not
     * Default value : false. The PCS is faster when it's set to false
     * Note : particle rotations are only applied to parent particles
     * Note : the particle rotations aren't stored values, so setting `computeParticleRotation` to false will prevents the particle to rotate
     */
    set computeParticleRotation(val) {
        this._computeParticleRotation = val;
    }
    /**
     * Tells to `setParticles()` to compute the particle colors or not.
     * Default value : true. The PCS is faster when it's set to false.
     * Note : the particle colors are stored values, so setting `computeParticleColor` to false will keep yet the last colors set.
     */
    set computeParticleColor(val) {
        this._computeParticleColor = val;
    }
    set computeParticleTexture(val) {
        this._computeParticleTexture = val;
    }
    /**
     * Gets if `setParticles()` computes the particle colors or not.
     * Default value : false. The PCS is faster when it's set to false.
     * Note : the particle colors are stored values, so setting `computeParticleColor` to false will keep yet the last colors set.
     */
    get computeParticleColor() {
        return this._computeParticleColor;
    }
    /**
     * Gets if `setParticles()` computes the particle textures or not.
     * Default value : false. The PCS is faster when it's set to false.
     * Note : the particle textures are stored values, so setting `computeParticleTexture` to false will keep yet the last colors set.
     */
    get computeParticleTexture() {
        return this._computeParticleTexture;
    }
    /**
     * Tells to `setParticles()` to compute or not the mesh bounding box when computing the particle positions.
     */
    set computeBoundingBox(val) {
        this._computeBoundingBox = val;
    }
    /**
     * Gets if `setParticles()` computes or not the mesh bounding box when computing the particle positions.
     */
    get computeBoundingBox() {
        return this._computeBoundingBox;
    }
    // =======================================================================
    // Particle behavior logic
    // these following methods may be overwritten by users to fit their needs
    /**
     * This function does nothing. It may be overwritten to set all the particle first values.
     * The PCS doesn't call this function, you may have to call it by your own.
     * doc :
     */
    initParticles() { }
    /**
     * This function does nothing. It may be overwritten to recycle a particle
     * The PCS doesn't call this function, you can to call it
     * doc :
     * @param particle The particle to recycle
     * @returns the recycled particle
     */
    recycleParticle(particle) {
        return particle;
    }
    /**
     * Updates a particle : this function should  be overwritten by the user.
     * It is called on each particle by `setParticles()`. This is the place to code each particle behavior.
     * doc :
     * @example : just set a particle position or velocity and recycle conditions
     * @param particle The particle to update
     * @returns the updated particle
     */
    updateParticle(particle) {
        return particle;
    }
    /**
     * This will be called before any other treatment by `setParticles()` and will be passed three parameters.
     * This does nothing and may be overwritten by the user.
     * @param start the particle index in the particle array where to start to iterate, same than the value passed to setParticle()
     * @param stop the particle index in the particle array where to stop to iterate, same than the value passed to setParticle()
     * @param update the boolean update value actually passed to setParticles()
     */
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    beforeUpdateParticles(start, stop, update) { }
    /**
     * This will be called  by `setParticles()` after all the other treatments and just before the actual mesh update.
     * This will be passed three parameters.
     * This does nothing and may be overwritten by the user.
     * @param start the particle index in the particle array where to start to iterate, same than the value passed to setParticle()
     * @param stop the particle index in the particle array where to stop to iterate, same than the value passed to setParticle()
     * @param update the boolean update value actually passed to setParticles()
     */
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    afterUpdateParticles(start, stop, update) { }
}

/* eslint-disable @typescript-eslint/promise-function-async */
/**
 * Parses SPZ data and returns a promise resolving to an IParsedPLY object.
 * @param data The ArrayBuffer containing SPZ data.
 * @param scene The Babylon.js scene.
 * @param loadingOptions Options for loading Gaussian Splatting files.
 * @returns A promise resolving to the parsed SPZ data.
 */
function ParseSpz(data, scene, loadingOptions) {
    const ubuf = new Uint8Array(data);
    const ubufu32 = new Uint32Array(data.slice(0, 12)); // Only need ubufu32[0] to [2]
    // debug infos
    const splatCount = ubufu32[2];
    const shDegree = ubuf[12];
    const fractionalBits = ubuf[13];
    const flags = ubuf[14];
    const reserved = ubuf[15];
    const version = ubufu32[1];
    // check magic and version
    if (reserved || ubufu32[0] != 0x5053474e || (version != 2 && version != 3)) {
        // reserved must be 0
        return new Promise((resolve) => {
            resolve({ mode: 3 /* Mode.Reject */, data: buffer, hasVertexColors: false });
        });
    }
    const rowOutputLength = 3 * 4 + 3 * 4 + 4 + 4; // 32
    const buffer = new ArrayBuffer(rowOutputLength * splatCount);
    const positionScale = 1.0 / (1 << fractionalBits);
    const int32View = new Int32Array(1);
    const uint8View = new Uint8Array(int32View.buffer);
    const read24bComponent = function (u8, offset) {
        uint8View[0] = u8[offset + 0];
        uint8View[1] = u8[offset + 1];
        uint8View[2] = u8[offset + 2];
        uint8View[3] = u8[offset + 2] & 0x80 ? 0xff : 0x00;
        return int32View[0] * positionScale;
    };
    let byteOffset = 16;
    const position = new Float32Array(buffer);
    const scale = new Float32Array(buffer);
    const rgba = new Uint8ClampedArray(buffer);
    const rot = new Uint8ClampedArray(buffer);
    // positions
    for (let i = 0; i < splatCount; i++) {
        position[i * 8 + 0] = read24bComponent(ubuf, byteOffset + 0);
        position[i * 8 + 1] = read24bComponent(ubuf, byteOffset + 3);
        position[i * 8 + 2] = read24bComponent(ubuf, byteOffset + 6);
        byteOffset += 9;
    }
    // colors
    const shC0 = 0.282;
    for (let i = 0; i < splatCount; i++) {
        for (let component = 0; component < 3; component++) {
            const byteValue = ubuf[byteOffset + splatCount + i * 3 + component];
            // 0.15 is hard coded value from spz
            // Scale factor for DC color components. To convert to RGB, we should multiply by 0.282, but it can
            // be useful to represent base colors that are out of range if the higher spherical harmonics bands
            // bring them back into range so we multiply by a smaller value.
            const value = (byteValue - 127.5) / (0.15 * 255);
            rgba[i * 32 + 24 + component] = Scalar.Clamp((0.5 + shC0 * value) * 255, 0, 255);
        }
        rgba[i * 32 + 24 + 3] = ubuf[byteOffset + i];
    }
    byteOffset += splatCount * 4;
    // scales
    for (let i = 0; i < splatCount; i++) {
        scale[i * 8 + 3 + 0] = Math.exp(ubuf[byteOffset + 0] / 16.0 - 10.0);
        scale[i * 8 + 3 + 1] = Math.exp(ubuf[byteOffset + 1] / 16.0 - 10.0);
        scale[i * 8 + 3 + 2] = Math.exp(ubuf[byteOffset + 2] / 16.0 - 10.0);
        byteOffset += 3;
    }
    // convert quaternion
    if (version >= 3) {
        /*
            In version 3, rotations are represented as the smallest three components of the normalized rotation quaternion, for optimal rotation accuracy.
            The largest component can be derived from the others and is not stored. Its index is stored on 2 bits
            and each of the smallest three components is encoded as a 10-bit signed integer.
        */
        const sqrt12 = Math.SQRT1_2;
        for (let i = 0; i < splatCount; i++) {
            const r = [ubuf[byteOffset + 0], ubuf[byteOffset + 1], ubuf[byteOffset + 2], ubuf[byteOffset + 3]];
            const comp = r[0] + (r[1] << 8) + (r[2] << 16) + (r[3] << 24);
            const cmask = (1 << 9) - 1;
            const rotation = [];
            const iLargest = comp >>> 30;
            let remaining = comp;
            let sumSquares = 0;
            for (let i = 3; i >= 0; --i) {
                if (i !== iLargest) {
                    const mag = remaining & cmask;
                    const negbit = (remaining >>> 9) & 0x1;
                    remaining = remaining >>> 10;
                    rotation[i] = sqrt12 * (mag / cmask);
                    if (negbit === 1) {
                        rotation[i] = -rotation[i];
                    }
                    // accumulate the sum of squares
                    sumSquares += rotation[i] * rotation[i];
                }
            }
            const square = 1 - sumSquares;
            rotation[iLargest] = Math.sqrt(Math.max(square, 0));
            const shuffle = [3, 0, 1, 2]; // shuffle to match the order of the quaternion components in the splat file
            for (let j = 0; j < 4; j++) {
                rot[i * 32 + 28 + j] = Math.round(127.5 + rotation[shuffle[j]] * 127.5);
            }
            byteOffset += 4;
        }
    }
    else {
        /*
            In version 2, rotations are represented as the `(x, y, z)` components of the normalized rotation quaternion. The
            `w` component can be derived from the others and is not stored. Each component is encoded as an
            8-bit signed integer.
        */
        for (let i = 0; i < splatCount; i++) {
            const x = ubuf[byteOffset + 0];
            const y = ubuf[byteOffset + 1];
            const z = ubuf[byteOffset + 2];
            const nx = x / 127.5 - 1;
            const ny = y / 127.5 - 1;
            const nz = z / 127.5 - 1;
            rot[i * 32 + 28 + 1] = x;
            rot[i * 32 + 28 + 2] = y;
            rot[i * 32 + 28 + 3] = z;
            const v = 1 - (nx * nx + ny * ny + nz * nz);
            rot[i * 32 + 28 + 0] = 127.5 + Math.sqrt(v < 0 ? 0 : v) * 127.5;
            byteOffset += 3;
        }
    }
    //SH
    if (shDegree) {
        // shVectorCount is : 3 for dim = 1, 8 for dim = 2 and 15 for dim = 3
        // number of vec3 vector needed per splat
        const shVectorCount = (shDegree + 1) * (shDegree + 1) - 1; // minus 1 because sh0 is color
        // number of component values : 3 per vector3 (45)
        const shComponentCount = shVectorCount * 3;
        const textureCount = Math.ceil(shComponentCount / 16); // 4 components can be stored per texture, 4 sh per component
        let shIndexRead = byteOffset;
        // sh is an array of uint8array that will be used to create sh textures
        const sh = [];
        const engine = scene.getEngine();
        const width = engine.getCaps().maxTextureSize;
        const height = Math.ceil(splatCount / width);
        // create array for the number of textures needed.
        for (let textureIndex = 0; textureIndex < textureCount; textureIndex++) {
            const texture = new Uint8Array(height * width * 4 * 4); // 4 components per texture, 4 sh per component
            sh.push(texture);
        }
        for (let i = 0; i < splatCount; i++) {
            for (let shIndexWrite = 0; shIndexWrite < shComponentCount; shIndexWrite++) {
                const shValue = ubuf[shIndexRead++];
                const textureIndex = Math.floor(shIndexWrite / 16);
                const shArray = sh[textureIndex];
                const byteIndexInTexture = shIndexWrite % 16; // [0..15]
                const offsetPerSplat = i * 16; // 16 sh values per texture per splat.
                shArray[byteIndexInTexture + offsetPerSplat] = shValue;
            }
        }
        return new Promise((resolve) => {
            resolve({ mode: 0 /* Mode.Splat */, data: buffer, hasVertexColors: false, sh: sh, trainedWithAntialiasing: !!flags });
        });
    }
    return new Promise((resolve) => {
        resolve({ mode: 0 /* Mode.Splat */, data: buffer, hasVertexColors: false, trainedWithAntialiasing: !!flags });
    });
}

const SH_C0 = 0.28209479177387814;
async function LoadWebpImageData(rootUrlOrData, filename, engine) {
    const promise = new Promise((resolve, reject) => {
        const image = engine.createCanvasImage();
        if (!image) {
            throw new Error("Failed to create ImageBitmap");
        }
        image.onload = () => {
            try {
                // Draw to canvas
                const canvas = engine.createCanvas(image.width, image.height);
                if (!canvas) {
                    throw new Error("Failed to create canvas");
                }
                const ctx = canvas.getContext("2d");
                if (!ctx) {
                    throw new Error("Failed to get 2D context");
                }
                ctx.drawImage(image, 0, 0);
                // Extract pixel data (RGBA per pixel)
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                resolve({ bits: new Uint8Array(imageData.data.buffer), width: imageData.width });
            }
            catch (error) {
                // eslint-disable-next-line @typescript-eslint/prefer-promise-reject-errors
                reject(`Error loading image ${image.src} with exception: ${error}`);
            }
        };
        image.onerror = (error) => {
            // eslint-disable-next-line @typescript-eslint/prefer-promise-reject-errors
            reject(`Error loading image ${image.src} with exception: ${error}`);
        };
        image.crossOrigin = "anonymous"; // To avoid CORS issues
        let objectUrl;
        if (typeof rootUrlOrData === "string") {
            // old behavior: URL + filename
            if (!filename) {
                throw new Error("filename is required when using a URL");
            }
            image.src = rootUrlOrData + filename;
        }
        else {
            // new behavior: Uint8Array
            const blob = new Blob([rootUrlOrData], { type: "image/webp" });
            objectUrl = URL.createObjectURL(blob);
            image.src = objectUrl;
        }
    });
    return await promise;
}
async function ParseSogDatas(data, imageDataArrays, scene) {
    const splatCount = data.count ? data.count : data.means.shape[0];
    const rowOutputLength = 3 * 4 + 3 * 4 + 4 + 4; // 32
    const buffer = new ArrayBuffer(rowOutputLength * splatCount);
    const position = new Float32Array(buffer);
    const scale = new Float32Array(buffer);
    const rgba = new Uint8ClampedArray(buffer);
    const rot = new Uint8ClampedArray(buffer);
    // Undo the symmetric log transform used at encode time:
    const unlog = (n) => Math.sign(n) * (Math.exp(Math.abs(n)) - 1);
    const meansl = imageDataArrays[0].bits;
    const meansu = imageDataArrays[1].bits;
    // Check that data.means.mins is an array
    if (!Array.isArray(data.means.mins) || !Array.isArray(data.means.maxs)) {
        throw new Error("Missing arrays in SOG data.");
    }
    // --- Positions
    for (let i = 0; i < splatCount; i++) {
        const index = i * 4;
        for (let j = 0; j < 3; j++) {
            const meansMin = data.means.mins[j];
            const meansMax = data.means.maxs[j];
            const meansup = meansu[index + j];
            const meanslow = meansl[index + j];
            const q = (meansup << 8) | meanslow;
            const n = Scalar.Lerp(meansMin, meansMax, q / 65535);
            position[i * 8 + j] = unlog(n);
        }
    }
    // --- Scales
    const scales = imageDataArrays[2].bits;
    if (data.version === 2) {
        if (!data.scales.codebook) {
            throw new Error("Missing codebook in SOG version 2 scales data.");
        }
        for (let i = 0; i < splatCount; i++) {
            const index = i * 4;
            for (let j = 0; j < 3; j++) {
                const sc = data.scales.codebook[scales[index + j]];
                const sce = Math.exp(sc);
                scale[i * 8 + 3 + j] = sce;
            }
        }
    }
    else {
        if (!Array.isArray(data.scales.mins) || !Array.isArray(data.scales.maxs)) {
            throw new Error("Missing arrays in SOG scales data.");
        }
        for (let i = 0; i < splatCount; i++) {
            const index = i * 4;
            for (let j = 0; j < 3; j++) {
                const sc = scales[index + j];
                const lsc = Scalar.Lerp(data.scales.mins[j], data.scales.maxs[j], sc / 255);
                const lsce = Math.exp(lsc);
                scale[i * 8 + 3 + j] = lsce;
            }
        }
    }
    // --- Colors/SH0
    const colors = imageDataArrays[4].bits;
    if (data.version === 2) {
        if (!data.sh0.codebook) {
            throw new Error("Missing codebook in SOG version 2 sh0 data.");
        }
        for (let i = 0; i < splatCount; i++) {
            const index = i * 4;
            for (let j = 0; j < 3; j++) {
                const component = 0.5 + data.sh0.codebook[colors[index + j]] * SH_C0;
                rgba[i * 32 + 24 + j] = Math.max(0, Math.min(255, Math.round(255 * component)));
            }
            rgba[i * 32 + 24 + 3] = colors[index + 3];
        }
    }
    else {
        if (!Array.isArray(data.sh0.mins) || !Array.isArray(data.sh0.maxs)) {
            throw new Error("Missing arrays in SOG sh0 data.");
        }
        for (let i = 0; i < splatCount; i++) {
            const index = i * 4;
            for (let j = 0; j < 4; j++) {
                const colorsMin = data.sh0.mins[j];
                const colorsMax = data.sh0.maxs[j];
                const colort = colors[index + j];
                const c = Scalar.Lerp(colorsMin, colorsMax, colort / 255);
                let csh;
                if (j < 3) {
                    csh = 0.5 + c * SH_C0;
                }
                else {
                    csh = 1.0 / (1.0 + Math.exp(-c));
                }
                rgba[i * 32 + 24 + j] = Math.max(0, Math.min(255, Math.round(255 * csh)));
            }
        }
    }
    // --- Rotations
    // Dequantize the stored three components:
    const toComp = (c) => ((c / 255 - 0.5) * 2.0) / Math.SQRT2;
    const quatArray = imageDataArrays[3].bits;
    for (let i = 0; i < splatCount; i++) {
        const quatsr = quatArray[i * 4 + 0];
        const quatsg = quatArray[i * 4 + 1];
        const quatsb = quatArray[i * 4 + 2];
        const quatsa = quatArray[i * 4 + 3];
        const a = toComp(quatsr);
        const b = toComp(quatsg);
        const c = toComp(quatsb);
        const mode = quatsa - 252; // 0..3 (R,G,B,A is one of the four components)
        // Reconstruct the omitted component so that ||q|| = 1 and w.l.o.g. the omitted one is non-negative
        const t = a * a + b * b + c * c;
        const d = Math.sqrt(Math.max(0, 1 - t));
        // Place components according to mode
        let q;
        switch (mode) {
            case 0:
                q = [d, a, b, c];
                break; // omitted = x
            case 1:
                q = [a, d, b, c];
                break; // omitted = y
            case 2:
                q = [a, b, d, c];
                break; // omitted = z
            case 3:
                q = [a, b, c, d];
                break; // omitted = w
            default:
                throw new Error("Invalid quaternion mode");
        }
        rot[i * 32 + 28 + 0] = q[0] * 127.5 + 127.5;
        rot[i * 32 + 28 + 1] = q[1] * 127.5 + 127.5;
        rot[i * 32 + 28 + 2] = q[2] * 127.5 + 127.5;
        rot[i * 32 + 28 + 3] = q[3] * 127.5 + 127.5;
    }
    // --- SH
    if (data.shN) {
        const coeffCounts = [0, 3, 8, 15];
        const coeffs = data.shN.bands ? coeffCounts[data.shN.bands] : data.shN.shape[1] / 3; // 3 components per coeff
        const shCentroids = imageDataArrays[5].bits;
        const shLabelsData = imageDataArrays[6].bits;
        const shCentroidsWidth = imageDataArrays[5].width;
        const shComponentCount = coeffs * 3;
        const textureCount = Math.ceil(shComponentCount / 16); // 4 components can be stored per texture, 4 sh per component
        //let shIndexRead = byteOffset;
        // sh is an array of uint8array that will be used to create sh textures
        const sh = [];
        const engine = scene.getEngine();
        const width = engine.getCaps().maxTextureSize;
        const height = Math.ceil(splatCount / width);
        // create array for the number of textures needed.
        for (let textureIndex = 0; textureIndex < textureCount; textureIndex++) {
            const texture = new Uint8Array(height * width * 4 * 4); // 4 components per texture, 4 sh per component
            sh.push(texture);
        }
        if (data.version === 2) {
            if (!data.shN.codebook) {
                throw new Error("Missing codebook in SOG version 2 shN data.");
            }
            for (let i = 0; i < splatCount; i++) {
                const n = shLabelsData[i * 4 + 0] + (shLabelsData[i * 4 + 1] << 8);
                const u = (n % 64) * coeffs;
                const v = Math.floor(n / 64);
                for (let k = 0; k < coeffs; k++) {
                    for (let j = 0; j < 3; j++) {
                        const shIndexWrite = k * 3 + j;
                        const textureIndex = Math.floor(shIndexWrite / 16);
                        const shArray = sh[textureIndex];
                        const byteIndexInTexture = shIndexWrite % 16; // [0..15]
                        const offsetPerSplat = i * 16; // 16 sh values per texture per splat.
                        const shValue = data.shN.codebook[shCentroids[(u + k) * 4 + j + v * shCentroidsWidth * 4]] * 127.5 + 127.5;
                        shArray[byteIndexInTexture + offsetPerSplat] = Math.max(0, Math.min(255, shValue));
                    }
                }
            }
        }
        else {
            for (let i = 0; i < splatCount; i++) {
                const n = shLabelsData[i * 4 + 0] + (shLabelsData[i * 4 + 1] << 8);
                const u = (n % 64) * coeffs;
                const v = Math.floor(n / 64);
                const shMin = data.shN.mins;
                const shMax = data.shN.maxs;
                for (let j = 0; j < 3; j++) {
                    for (let k = 0; k < coeffs / 3; k++) {
                        const shIndexWrite = k * 3 + j;
                        const textureIndex = Math.floor(shIndexWrite / 16);
                        const shArray = sh[textureIndex];
                        const byteIndexInTexture = shIndexWrite % 16; // [0..15]
                        const offsetPerSplat = i * 16; // 16 sh values per texture per splat.
                        const shValue = Scalar.Lerp(shMin, shMax, shCentroids[(u + k) * 4 + j + v * shCentroidsWidth * 4] / 255) * 127.5 + 127.5;
                        shArray[byteIndexInTexture + offsetPerSplat] = Math.max(0, Math.min(255, shValue));
                    }
                }
            }
        }
        return await new Promise((resolve) => {
            resolve({ mode: 0 /* Mode.Splat */, data: buffer, hasVertexColors: false, sh: sh });
        });
    }
    return await new Promise((resolve) => {
        resolve({ mode: 0 /* Mode.Splat */, data: buffer, hasVertexColors: false });
    });
}
/**
 * Parse SOG data from either a SOGRootData object (with webp files loaded from rootUrl) or from a Map of filenames to Uint8Array file data (including meta.json)
 * @param dataOrFiles Either the SOGRootData or a Map of filenames to Uint8Array file data (including meta.json)
 * @param rootUrl Base URL to load webp files from (if dataOrFiles is SOGRootData)
 * @param scene The Babylon.js scene
 * @returns Parsed data
 */
async function ParseSogMeta(dataOrFiles, rootUrl, scene) {
    let data;
    let files;
    if (dataOrFiles instanceof Map) {
        files = dataOrFiles;
        const metaFile = files.get("meta.json");
        if (!metaFile) {
            throw new Error("meta.json not found in files Map");
        }
        data = JSON.parse(new TextDecoder().decode(metaFile));
    }
    else {
        data = dataOrFiles;
    }
    // Collect all file names
    const urls = [...data.means.files, ...data.scales.files, ...data.quats.files, ...data.sh0.files];
    if (data.shN) {
        urls.push(...data.shN.files);
    }
    // Load webp images in parallel
    const imageDataArrays = await Promise.all(urls.map(async (fileName) => {
        if (files && files.has(fileName)) {
            // load from in-memory Uint8Array
            const fileData = files.get(fileName);
            return await LoadWebpImageData(fileData, fileName, scene.getEngine());
        }
        else {
            // fallback: load from URL
            return await LoadWebpImageData(rootUrl, fileName, scene.getEngine());
        }
    }));
    return await ParseSogDatas(data, imageDataArrays, scene);
}

/**
 * @experimental
 * SPLAT file type loader.
 * This is a babylon scene loader plugin.
 */
class SPLATFileLoader {
    /**
     * Creates loader for gaussian splatting files
     * @param loadingOptions options for loading and parsing splat and PLY files.
     */
    constructor(loadingOptions = SPLATFileLoader._DefaultLoadingOptions) {
        /**
         * Defines the name of the plugin.
         */
        this.name = SPLATFileLoaderMetadata.name;
        this._assetContainer = null;
        /**
         * Defines the extensions the splat loader is able to load.
         * force data to come in as an ArrayBuffer
         */
        this.extensions = SPLATFileLoaderMetadata.extensions;
        this._loadingOptions = loadingOptions;
    }
    /** @internal */
    createPlugin(options) {
        return new SPLATFileLoader(options[SPLATFileLoaderMetadata.name]);
    }
    /**
     * Imports  from the loaded gaussian splatting data and adds them to the scene
     * @param meshesNames a string or array of strings of the mesh names that should be loaded from the file
     * @param scene the scene the meshes should be added to
     * @param data the gaussian splatting data to load
     * @param rootUrl root url to load from
     * @param _onProgress callback called while file is loading
     * @param _fileName Defines the name of the file to load
     * @returns a promise containing the loaded meshes, particles, skeletons and animations
     */
    async importMeshAsync(meshesNames, scene, data, rootUrl, _onProgress, _fileName) {
        // eslint-disable-next-line github/no-then
        return await this._parseAsync(meshesNames, scene, data, rootUrl).then((meshes) => {
            return {
                meshes: meshes,
                particleSystems: [],
                skeletons: [],
                animationGroups: [],
                transformNodes: [],
                geometries: [],
                lights: [],
                spriteManagers: [],
            };
        });
    }
    static _BuildPointCloud(pointcloud, data) {
        if (!data.byteLength) {
            return false;
        }
        const uBuffer = new Uint8Array(data);
        const fBuffer = new Float32Array(data);
        // parsed array contains room for position(3floats), normal(3floats), color (4b), quantized quaternion (4b)
        const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
        const vertexCount = uBuffer.length / rowLength;
        const pointcloudfunc = function (particle, i) {
            const x = fBuffer[8 * i + 0];
            const y = fBuffer[8 * i + 1];
            const z = fBuffer[8 * i + 2];
            particle.position = new Vector3(x, y, z);
            const r = uBuffer[rowLength * i + 24 + 0] / 255;
            const g = uBuffer[rowLength * i + 24 + 1] / 255;
            const b = uBuffer[rowLength * i + 24 + 2] / 255;
            particle.color = new Color4(r, g, b, 1);
        };
        pointcloud.addPoints(vertexCount, pointcloudfunc);
        return true;
    }
    static _BuildMesh(scene, parsedPLY) {
        const mesh = new Mesh("PLYMesh", scene);
        const uBuffer = new Uint8Array(parsedPLY.data);
        const fBuffer = new Float32Array(parsedPLY.data);
        const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
        const vertexCount = uBuffer.length / rowLength;
        const positions = [];
        const vertexData = new VertexData();
        for (let i = 0; i < vertexCount; i++) {
            const x = fBuffer[8 * i + 0];
            const y = fBuffer[8 * i + 1];
            const z = fBuffer[8 * i + 2];
            positions.push(x, y, z);
        }
        if (parsedPLY.hasVertexColors) {
            const colors = new Float32Array(vertexCount * 4);
            for (let i = 0; i < vertexCount; i++) {
                const r = uBuffer[rowLength * i + 24 + 0] / 255;
                const g = uBuffer[rowLength * i + 24 + 1] / 255;
                const b = uBuffer[rowLength * i + 24 + 2] / 255;
                colors[i * 4 + 0] = r;
                colors[i * 4 + 1] = g;
                colors[i * 4 + 2] = b;
                colors[i * 4 + 3] = 1;
            }
            vertexData.colors = colors;
        }
        vertexData.positions = positions;
        vertexData.indices = parsedPLY.faces;
        vertexData.applyToMesh(mesh);
        return mesh;
    }
    // eslint-disable-next-line @typescript-eslint/promise-function-async, no-restricted-syntax, @typescript-eslint/naming-convention
    async _unzipWithFFlateAsync(data) {
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        let fflate = this._loadingOptions.fflate;
        // ensure fflate is loaded
        if (!fflate) {
            if (typeof window.fflate === "undefined") {
                await Tools.LoadScriptAsync(this._loadingOptions.deflateURL ?? "https://unpkg.com/fflate/umd/index.js");
            }
            fflate = window.fflate;
        }
        const { unzipSync } = fflate;
        const unzipped = unzipSync(data); // { [filename: string]: Uint8Array }
        const files = new Map();
        for (const [name, content] of Object.entries(unzipped)) {
            files.set(name, content);
        }
        return files;
    }
    // eslint-disable-next-line @typescript-eslint/promise-function-async, no-restricted-syntax
    _parseAsync(meshesNames, scene, data, rootUrl) {
        const babylonMeshesArray = []; //The mesh for babylon
        const makeGSFromParsedSOG = (parsedSOG) => {
            scene._blockEntityCollection = !!this._assetContainer;
            const gaussianSplatting = this._loadingOptions.gaussianSplattingMesh ?? new GaussianSplattingMesh("GaussianSplatting", null, scene, this._loadingOptions.keepInRam);
            gaussianSplatting._parentContainer = this._assetContainer;
            babylonMeshesArray.push(gaussianSplatting);
            gaussianSplatting.updateData(parsedSOG.data, parsedSOG.sh, { flipY: false });
            gaussianSplatting.scaling.y *= -1;
            gaussianSplatting.computeWorldMatrix(true);
            scene._blockEntityCollection = false;
        };
        // check if data is json string
        if (typeof data === "string") {
            const dataSOG = JSON.parse(data);
            if (dataSOG && dataSOG.means && dataSOG.scales && dataSOG.quats && dataSOG.sh0) {
                return new Promise((resolve) => {
                    ParseSogMeta(dataSOG, rootUrl, scene)
                        // eslint-disable-next-line @typescript-eslint/no-floating-promises, github/no-then
                        .then((parsedSOG) => {
                        makeGSFromParsedSOG(parsedSOG);
                        resolve(babylonMeshesArray);
                    })
                        // eslint-disable-next-line github/no-then
                        .catch(() => {
                        throw new Error("Failed to parse SOG data.");
                    });
                });
            }
        }
        const u8 = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
        // ZIP signature check for SOG
        if (u8[0] === 0x50 && u8[1] === 0x4b) {
            return new Promise((resolve) => {
                // eslint-disable-next-line @typescript-eslint/no-floating-promises, github/no-then
                this._unzipWithFFlateAsync(u8).then((files) => {
                    ParseSogMeta(files, rootUrl, scene)
                        // eslint-disable-next-line @typescript-eslint/no-floating-promises, github/no-then
                        .then((parsedSOG) => {
                        makeGSFromParsedSOG(parsedSOG);
                        resolve(babylonMeshesArray);
                    }) // eslint-disable-next-line github/no-then
                        .catch(() => {
                        throw new Error("Failed to parse SOG zip data.");
                    });
                });
            });
        }
        const readableStream = new ReadableStream({
            start(controller) {
                controller.enqueue(new Uint8Array(data)); // Enqueue the ArrayBuffer as a Uint8Array
                controller.close();
            },
        });
        // Use GZip DecompressionStream
        const decompressionStream = new DecompressionStream("gzip");
        const decompressedStream = readableStream.pipeThrough(decompressionStream);
        return new Promise((resolve) => {
            new Response(decompressedStream)
                .arrayBuffer()
                // eslint-disable-next-line github/no-then
                .then((buffer) => {
                // eslint-disable-next-line @typescript-eslint/no-floating-promises, github/no-then
                ParseSpz(buffer, scene, this._loadingOptions).then((parsedSPZ) => {
                    scene._blockEntityCollection = !!this._assetContainer;
                    const gaussianSplatting = this._loadingOptions.gaussianSplattingMesh ?? new GaussianSplattingMesh("GaussianSplatting", null, scene, this._loadingOptions.keepInRam);
                    if (parsedSPZ.trainedWithAntialiasing) {
                        const gsMaterial = gaussianSplatting.material;
                        gsMaterial.kernelSize = 0.1;
                        gsMaterial.compensation = true;
                    }
                    gaussianSplatting._parentContainer = this._assetContainer;
                    babylonMeshesArray.push(gaussianSplatting);
                    gaussianSplatting.updateData(parsedSPZ.data, parsedSPZ.sh, { flipY: false });
                    if (!this._loadingOptions.flipY) {
                        gaussianSplatting.scaling.y *= -1;
                        gaussianSplatting.computeWorldMatrix(true);
                    }
                    scene._blockEntityCollection = false;
                    this.applyAutoCameraLimits(parsedSPZ, scene);
                    resolve(babylonMeshesArray);
                });
            })
                // eslint-disable-next-line github/no-then
                .catch(() => {
                // Catch any decompression errors
                // eslint-disable-next-line @typescript-eslint/no-floating-promises, github/no-then
                SPLATFileLoader._ConvertPLYToSplat(data).then(async (parsedPLY) => {
                    scene._blockEntityCollection = !!this._assetContainer;
                    switch (parsedPLY.mode) {
                        case 0 /* Mode.Splat */:
                            {
                                const gaussianSplatting = this._loadingOptions.gaussianSplattingMesh ?? new GaussianSplattingMesh("GaussianSplatting", null, scene, this._loadingOptions.keepInRam);
                                gaussianSplatting._parentContainer = this._assetContainer;
                                babylonMeshesArray.push(gaussianSplatting);
                                gaussianSplatting.updateData(parsedPLY.data, parsedPLY.sh, { flipY: false });
                                gaussianSplatting.scaling.y *= -1;
                                if (parsedPLY.chirality === "RightHanded") {
                                    gaussianSplatting.scaling.y *= -1;
                                }
                                switch (parsedPLY.upAxis) {
                                    case "X":
                                        gaussianSplatting.rotation = new Vector3(0, 0, Math.PI / 2);
                                        break;
                                    case "Y":
                                        gaussianSplatting.rotation = new Vector3(0, 0, Math.PI);
                                        break;
                                    case "Z":
                                        gaussianSplatting.rotation = new Vector3(-Math.PI / 2, Math.PI, 0);
                                        break;
                                }
                                gaussianSplatting.computeWorldMatrix(true);
                            }
                            break;
                        case 1 /* Mode.PointCloud */:
                            {
                                const pointcloud = new PointsCloudSystem("PointCloud", 1, scene);
                                if (SPLATFileLoader._BuildPointCloud(pointcloud, parsedPLY.data)) {
                                    // eslint-disable-next-line github/no-then
                                    await pointcloud.buildMeshAsync().then((mesh) => {
                                        babylonMeshesArray.push(mesh);
                                    });
                                }
                                else {
                                    pointcloud.dispose();
                                }
                            }
                            break;
                        case 2 /* Mode.Mesh */:
                            {
                                if (parsedPLY.faces) {
                                    babylonMeshesArray.push(SPLATFileLoader._BuildMesh(scene, parsedPLY));
                                }
                                else {
                                    throw new Error("PLY mesh doesn't contain face informations.");
                                }
                            }
                            break;
                        default:
                            throw new Error("Unsupported Splat mode");
                    }
                    scene._blockEntityCollection = false;
                    this.applyAutoCameraLimits(parsedPLY, scene);
                    resolve(babylonMeshesArray);
                });
            });
        });
    }
    /**
     * Applies camera limits based on parsed meta data
     * @param meta parsed splat meta data
     * @param scene
     */
    applyAutoCameraLimits(meta, scene) {
        if (this._loadingOptions.disableAutoCameraLimits) {
            return;
        }
        if ((meta.safeOrbitCameraRadiusMin !== undefined || meta.safeOrbitCameraElevationMinMax !== undefined) && scene.activeCamera?.getClassName() === "ArcRotateCamera") {
            const arcCam = scene.activeCamera;
            if (meta.safeOrbitCameraElevationMinMax) {
                arcCam.lowerBetaLimit = Math.PI * 0.5 - meta.safeOrbitCameraElevationMinMax[1];
                arcCam.upperBetaLimit = Math.PI * 0.5 - meta.safeOrbitCameraElevationMinMax[0];
            }
            if (meta.safeOrbitCameraRadiusMin) {
                arcCam.lowerRadiusLimit = meta.safeOrbitCameraRadiusMin;
            }
        }
    }
    /**
     * Load into an asset container.
     * @param scene The scene to load into
     * @param data The data to import
     * @param rootUrl The root url for scene and resources
     * @returns The loaded asset container
     */
    // eslint-disable-next-line no-restricted-syntax
    loadAssetContainerAsync(scene, data, rootUrl) {
        const container = new AssetContainer(scene);
        this._assetContainer = container;
        return (this.importMeshAsync(null, scene, data, rootUrl)
            // eslint-disable-next-line github/no-then
            .then((result) => {
            for (const mesh of result.meshes) {
                container.meshes.push(mesh);
            }
            // mesh material will be null before 1st rendered frame.
            this._assetContainer = null;
            return container;
        })
            // eslint-disable-next-line github/no-then
            .catch((ex) => {
            this._assetContainer = null;
            throw ex;
        }));
    }
    /**
     * Imports all objects from the loaded OBJ data and adds them to the scene
     * @param scene the scene the objects should be added to
     * @param data the OBJ data to load
     * @param rootUrl root url to load from
     * @returns a promise which completes when objects have been loaded to the scene
     */
    // eslint-disable-next-line @typescript-eslint/promise-function-async, no-restricted-syntax
    loadAsync(scene, data, rootUrl) {
        //Get the 3D model
        // eslint-disable-next-line github/no-then
        return this.importMeshAsync(null, scene, data, rootUrl).then(() => {
            // return void
        });
    }
    /**
     * Code from https://github.com/dylanebert/gsplat.js/blob/main/src/loaders/PLYLoader.ts Under MIT license
     * Converts a .ply data array buffer to splat
     * if data array buffer is not ply, returns the original buffer
     * @param data the .ply data to load
     * @returns the loaded splat buffer
     */
    static _ConvertPLYToSplat(data) {
        const ubuf = new Uint8Array(data);
        const header = new TextDecoder().decode(ubuf.slice(0, 1024 * 10));
        const headerEnd = "end_header\n";
        const headerEndIndex = header.indexOf(headerEnd);
        if (headerEndIndex < 0 || !header) {
            // standard splat
            return new Promise((resolve) => {
                resolve({ mode: 0 /* Mode.Splat */, data: data, rawSplat: true });
            });
        }
        const vertexCount = parseInt(/element vertex (\d+)\n/.exec(header)[1]);
        const faceElement = /element face (\d+)\n/.exec(header);
        let faceCount = 0;
        if (faceElement) {
            faceCount = parseInt(faceElement[1]);
        }
        const chunkElement = /element chunk (\d+)\n/.exec(header);
        let chunkCount = 0;
        if (chunkElement) {
            chunkCount = parseInt(chunkElement[1]);
        }
        let rowVertexOffset = 0;
        let rowChunkOffset = 0;
        const offsets = {
            double: 8,
            int: 4,
            uint: 4,
            float: 4,
            short: 2,
            ushort: 2,
            uchar: 1,
            list: 0,
        };
        const ElementMode = {
            Vertex: 0,
            Chunk: 1,
            SH: 2,
            Float_Tuple: 3,
            Float: 4,
            Uchar: 5,
        };
        let chunkMode = ElementMode.Chunk;
        const vertexProperties = [];
        const filtered = header.slice(0, headerEndIndex).split("\n");
        const metaData = {};
        for (const prop of filtered) {
            if (prop.startsWith("property ")) {
                const [, type, name] = prop.split(" ");
                if (chunkMode == ElementMode.Chunk) {
                    rowChunkOffset += offsets[type];
                }
                else if (chunkMode == ElementMode.Vertex) {
                    vertexProperties.push({ name, type, offset: rowVertexOffset });
                    rowVertexOffset += offsets[type];
                }
                else if (chunkMode == ElementMode.SH) {
                    vertexProperties.push({ name, type, offset: rowVertexOffset });
                }
                else if (chunkMode == ElementMode.Float_Tuple) {
                    const view = new DataView(data, rowChunkOffset, offsets.float * 2);
                    metaData.safeOrbitCameraElevationMinMax = [view.getFloat32(0, true), view.getFloat32(4, true)];
                }
                else if (chunkMode == ElementMode.Float) {
                    const view = new DataView(data, rowChunkOffset, offsets.float);
                    metaData.safeOrbitCameraRadiusMin = view.getFloat32(0, true);
                }
                else if (chunkMode == ElementMode.Uchar) {
                    const view = new DataView(data, rowChunkOffset, offsets.uchar);
                    if (name == "up_axis") {
                        metaData.upAxis = view.getUint8(0) == 0 ? "X" : view.getUint8(0) == 1 ? "Y" : "Z";
                    }
                    else if (name == "chirality") {
                        metaData.chirality = view.getUint8(0) == 0 ? "LeftHanded" : "RightHanded";
                    }
                }
                if (!offsets[type]) {
                    Logger.Warn(`Unsupported property type: ${type}.`);
                }
            }
            else if (prop.startsWith("element ")) {
                const [, type] = prop.split(" ");
                if (type == "chunk") {
                    chunkMode = ElementMode.Chunk;
                }
                else if (type == "vertex") {
                    chunkMode = ElementMode.Vertex;
                }
                else if (type == "sh") {
                    chunkMode = ElementMode.SH;
                }
                else if (type == "safe_orbit_camera_elevation_min_max_radians") {
                    chunkMode = ElementMode.Float_Tuple;
                }
                else if (type == "safe_orbit_camera_radius_min") {
                    chunkMode = ElementMode.Float;
                }
                else if (type == "up_axis" || type == "chirality") {
                    chunkMode = ElementMode.Uchar;
                }
            }
        }
        const rowVertexLength = rowVertexOffset;
        const rowChunkLength = rowChunkOffset;
        // eslint-disable-next-line github/no-then
        return GaussianSplattingMesh.ConvertPLYWithSHToSplatAsync(data).then(async (splatsData) => {
            const dataView = new DataView(data, headerEndIndex + headerEnd.length);
            let offset = rowChunkLength * chunkCount + rowVertexLength * vertexCount;
            // faces
            const faces = [];
            if (faceCount) {
                for (let i = 0; i < faceCount; i++) {
                    const faceVertexCount = dataView.getUint8(offset);
                    if (faceVertexCount != 3) {
                        continue; // only support triangles
                    }
                    offset += 1;
                    for (let j = 0; j < faceVertexCount; j++) {
                        const vertexIndex = dataView.getUint32(offset + (2 - j) * 4, true); // change face winding
                        faces.push(vertexIndex);
                    }
                    offset += 12;
                }
            }
            // early exit for chunked/quantized ply
            if (chunkCount) {
                return await new Promise((resolve) => {
                    resolve({ mode: 0 /* Mode.Splat */, data: splatsData.buffer, sh: splatsData.sh, faces: faces, hasVertexColors: false, compressed: true, rawSplat: false });
                });
            }
            // count available properties. if all necessary are present then it's a splat. Otherwise, it's a point cloud
            // if faces are found, then it's a standard mesh
            let propertyCount = 0;
            let propertyColorCount = 0;
            const splatProperties = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity", "rot_0", "rot_1", "rot_2", "rot_3"];
            const splatColorProperties = ["red", "green", "blue", "f_dc_0", "f_dc_1", "f_dc_2"];
            for (let propertyIndex = 0; propertyIndex < vertexProperties.length; propertyIndex++) {
                const property = vertexProperties[propertyIndex];
                if (splatProperties.includes(property.name)) {
                    propertyCount++;
                }
                if (splatColorProperties.includes(property.name)) {
                    propertyColorCount++;
                }
            }
            const hasMandatoryProperties = propertyCount == splatProperties.length && propertyColorCount == 3;
            const currentMode = faceCount ? 2 /* Mode.Mesh */ : hasMandatoryProperties ? 0 /* Mode.Splat */ : 1 /* Mode.PointCloud */;
            // parsed ready ready to be used as a splat
            return await new Promise((resolve) => {
                resolve({
                    ...metaData,
                    mode: currentMode,
                    data: splatsData.buffer,
                    sh: splatsData.sh,
                    faces: faces,
                    hasVertexColors: !!propertyColorCount,
                    compressed: false,
                    rawSplat: false,
                });
            });
        });
    }
}
SPLATFileLoader._DefaultLoadingOptions = {
    keepInRam: false,
    flipY: false,
};
// Add this loader into the register plugin
RegisterSceneLoaderPlugin(new SPLATFileLoader());

export { SPLATFileLoader };
//# sourceMappingURL=splatFileLoader-YCHqOTfX.js.map
