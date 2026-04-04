const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["./rgbdDecode.fragment-DwPrei9n.js","./index-CxR0Yf_7.js","./index-BvBk1Iap.js","./index-CrbFh6qt.css","./helperFunctions-qLR1vvLh.js","./rgbdDecode.fragment-PHs4X25h.js","./helperFunctions-CiFVmLcr.js"])))=>i.map(i=>d[i]);
import { af as __vitePreload } from './index-BvBk1Iap.js';
import { L as Logger, ay as SphericalPolynomial, V as Vector3, I as InternalTexture, B as BaseTexture, d as Tools, az as ILog2, aA as PostProcess } from './index-CxR0Yf_7.js';
import './dumpTools-BrtW8jV6.js';

const DefaultEnvironmentTextureImageType = "image/png";
const CurrentVersion = 2;
/**
 * Magic number identifying the env file.
 */
const MagicBytes = [0x86, 0x16, 0x87, 0x96, 0xf6, 0xd6, 0x96, 0x36];
/**
 * Gets the environment info from an env file.
 * @param data The array buffer containing the .env bytes.
 * @returns the environment file info (the json header) if successfully parsed, normalized to the latest supported version.
 */
function GetEnvInfo(data) {
    const dataView = new DataView(data.buffer, data.byteOffset, data.byteLength);
    let pos = 0;
    for (let i = 0; i < MagicBytes.length; i++) {
        if (dataView.getUint8(pos++) !== MagicBytes[i]) {
            Logger.Error("Not a babylon environment map");
            return null;
        }
    }
    // Read json manifest - collect characters up to null terminator
    let manifestString = "";
    let charCode = 0x00;
    while ((charCode = dataView.getUint8(pos++))) {
        manifestString += String.fromCharCode(charCode);
    }
    let manifest = JSON.parse(manifestString);
    manifest = normalizeEnvInfo(manifest);
    // Extend the header with the position of the payload.
    manifest.binaryDataPosition = pos;
    if (manifest.specular) {
        // Fallback to 0.8 exactly if lodGenerationScale is not defined for backward compatibility.
        manifest.specular.lodGenerationScale = manifest.specular.lodGenerationScale || 0.8;
    }
    return manifest;
}
/**
 * Normalizes any supported version of the environment file info to the latest version
 * @param info environment file info on any supported version
 * @returns environment file info in the latest supported version
 * @private
 */
function normalizeEnvInfo(info) {
    if (info.version > CurrentVersion) {
        throw new Error(`Unsupported babylon environment map version "${info.version}". Latest supported version is "${CurrentVersion}".`);
    }
    if (info.version === 2) {
        return info;
    }
    // Migrate a v1 info to v2
    info = { ...info, version: 2, imageType: DefaultEnvironmentTextureImageType };
    return info;
}
/**
 * Creates the ArrayBufferViews used for initializing environment texture image data.
 * @param data the image data
 * @param info parameters that determine what views will be created for accessing the underlying buffer
 * @returns the views described by info providing access to the underlying buffer
 */
function CreateRadianceImageDataArrayBufferViews(data, info) {
    info = normalizeEnvInfo(info);
    const specularInfo = info.specular;
    // Double checks the enclosed info
    let mipmapsCount = Math.log2(info.width);
    mipmapsCount = Math.round(mipmapsCount) + 1;
    if (specularInfo.mipmaps.length !== 6 * mipmapsCount) {
        throw new Error(`Unsupported specular mipmaps number "${specularInfo.mipmaps.length}"`);
    }
    const imageData = new Array(mipmapsCount);
    for (let i = 0; i < mipmapsCount; i++) {
        imageData[i] = new Array(6);
        for (let face = 0; face < 6; face++) {
            const imageInfo = specularInfo.mipmaps[i * 6 + face];
            imageData[i][face] = new Uint8Array(data.buffer, data.byteOffset + info.binaryDataPosition + imageInfo.position, imageInfo.length);
        }
    }
    return imageData;
}
/**
 * Creates the ArrayBufferViews used for initializing environment texture image data.
 * @param data the image data
 * @param info parameters that determine what views will be created for accessing the underlying buffer
 * @returns the views described by info providing access to the underlying buffer
 */
function CreateIrradianceImageDataArrayBufferViews(data, info) {
    info = normalizeEnvInfo(info);
    const imageData = new Array(6);
    const irradianceTexture = info.irradiance?.irradianceTexture;
    if (irradianceTexture) {
        if (irradianceTexture.faces.length !== 6) {
            throw new Error(`Incorrect irradiance texture faces number "${irradianceTexture.faces.length}"`);
        }
        for (let face = 0; face < 6; face++) {
            const imageInfo = irradianceTexture.faces[face];
            imageData[face] = new Uint8Array(data.buffer, data.byteOffset + info.binaryDataPosition + imageInfo.position, imageInfo.length);
        }
    }
    return imageData;
}
/**
 * Uploads the texture info contained in the env file to the GPU.
 * @param texture defines the internal texture to upload to
 * @param data defines the data to load
 * @param info defines the texture info retrieved through the GetEnvInfo method
 * @returns a promise
 */
// eslint-disable-next-line @typescript-eslint/promise-function-async, no-restricted-syntax
function UploadEnvLevelsAsync(texture, data, info) {
    info = normalizeEnvInfo(info);
    const specularInfo = info.specular;
    if (!specularInfo) {
        // Nothing else parsed so far
        return Promise.resolve([]);
    }
    texture._lodGenerationScale = specularInfo.lodGenerationScale;
    const promises = [];
    const radianceImageData = CreateRadianceImageDataArrayBufferViews(data, info);
    promises.push(UploadRadianceLevelsAsync(texture, radianceImageData, info.imageType));
    const irradianceTexture = info.irradiance?.irradianceTexture;
    if (irradianceTexture) {
        const irradianceImageData = CreateIrradianceImageDataArrayBufferViews(data, info);
        let dominantDirection = null;
        if (info.irradiance?.irradianceTexture?.dominantDirection) {
            dominantDirection = Vector3.FromArray(info.irradiance.irradianceTexture.dominantDirection);
        }
        promises.push(UploadIrradianceLevelsAsync(texture, irradianceImageData, irradianceTexture.size, info.imageType, dominantDirection));
    }
    return Promise.all(promises);
}
async function _OnImageReadyAsync(image, engine, expandTexture, rgbdPostProcess, url, face, i, generateNonLODTextures, lodTextures, cubeRtt, texture) {
    return await new Promise((resolve, reject) => {
        if (expandTexture) {
            const tempTexture = engine.createTexture(null, true, true, null, 1, null, (message) => {
                // eslint-disable-next-line @typescript-eslint/prefer-promise-reject-errors
                reject(message);
            }, image);
            rgbdPostProcess?.onEffectCreatedObservable.addOnce((effect) => {
                effect.executeWhenCompiled(() => {
                    // Uncompress the data to a RTT
                    rgbdPostProcess.externalTextureSamplerBinding = true;
                    rgbdPostProcess.onApply = (effect) => {
                        effect._bindTexture("textureSampler", tempTexture);
                        effect.setFloat2("scale", 1, engine._features.needsInvertingBitmap && image instanceof ImageBitmap ? -1 : 1);
                    };
                    if (!engine.scenes.length) {
                        return;
                    }
                    engine.scenes[0].postProcessManager.directRender([rgbdPostProcess], cubeRtt, true, face, i);
                    // Cleanup
                    engine.restoreDefaultFramebuffer();
                    tempTexture.dispose();
                    URL.revokeObjectURL(url);
                    resolve();
                });
            });
        }
        else {
            engine._uploadImageToTexture(texture, image, face, i);
            // Upload the face to the non lod texture support
            if (generateNonLODTextures) {
                const lodTexture = lodTextures[i];
                if (lodTexture) {
                    engine._uploadImageToTexture(lodTexture._texture, image, face, 0);
                }
            }
            resolve();
        }
    });
}
/**
 * Uploads the levels of image data to the GPU.
 * @param texture defines the internal texture to upload to
 * @param imageData defines the array buffer views of image data [mipmap][face]
 * @param imageType the mime type of the image data
 * @returns a promise
 */
async function UploadRadianceLevelsAsync(texture, imageData, imageType = DefaultEnvironmentTextureImageType) {
    const engine = texture.getEngine();
    texture.format = 5;
    texture.type = 0;
    texture.generateMipMaps = true;
    texture._cachedAnisotropicFilteringLevel = null;
    engine.updateTextureSamplingMode(3, texture);
    await _UploadLevelsAsync(texture, imageData, true, imageType);
    // Flag internal texture as ready in case they are in use.
    texture.isReady = true;
}
/**
 * Uploads the levels of image data to the GPU.
 * @param mainTexture defines the internal texture to upload to
 * @param imageData defines the array buffer views of image data [mipmap][face]
 * @param size defines the size of the texture faces
 * @param imageType the mime type of the image data
 * @param dominantDirection the dominant direction of light in the environment texture, if available
 * @returns a promise
 */
async function UploadIrradianceLevelsAsync(mainTexture, imageData, size, imageType = DefaultEnvironmentTextureImageType, dominantDirection = null) {
    // Gets everything ready.
    const engine = mainTexture.getEngine();
    const texture = new InternalTexture(engine, 5 /* InternalTextureSource.RenderTarget */);
    const baseTexture = new BaseTexture(engine, texture);
    mainTexture._irradianceTexture = baseTexture;
    baseTexture._dominantDirection = dominantDirection;
    texture.isCube = true;
    texture.format = 5;
    texture.type = 0;
    texture.generateMipMaps = true;
    texture._cachedAnisotropicFilteringLevel = null;
    texture.generateMipMaps = true;
    texture.width = size;
    texture.height = size;
    engine.updateTextureSamplingMode(3, texture);
    await _UploadLevelsAsync(texture, [imageData], false, imageType);
    engine.generateMipMapsForCubemap(texture);
    // Flag internal texture as ready in case they are in use.
    texture.isReady = true;
}
/**
 * Uploads the levels of image data to the GPU.
 * @param texture defines the internal texture to upload to
 * @param imageData defines the array buffer views of image data [mipmap][face]
 * @param canGenerateNonLODTextures defines whether or not to generate non lod textures
 * @param imageType the mime type of the image data
 * @returns a promise
 */
async function _UploadLevelsAsync(texture, imageData, canGenerateNonLODTextures, imageType = DefaultEnvironmentTextureImageType) {
    if (!Tools.IsExponentOfTwo(texture.width)) {
        throw new Error("Texture size must be a power of two");
    }
    const mipmapsCount = ILog2(texture.width) + 1;
    // Gets everything ready.
    const engine = texture.getEngine();
    let expandTexture = false;
    let generateNonLODTextures = false;
    let rgbdPostProcess = null;
    let cubeRtt = null;
    let lodTextures = null;
    const caps = engine.getCaps();
    if (!caps.textureLOD) {
        expandTexture = false;
        generateNonLODTextures = canGenerateNonLODTextures;
    }
    else if (!engine._features.supportRenderAndCopyToLodForFloatTextures) {
        expandTexture = false;
    }
    // If half float available we can uncompress the texture
    else if (caps.textureHalfFloatRender && caps.textureHalfFloatLinearFiltering) {
        expandTexture = true;
        texture.type = 2;
    }
    // If full float available we can uncompress the texture
    else if (caps.textureFloatRender && caps.textureFloatLinearFiltering) {
        expandTexture = true;
        texture.type = 1;
    }
    // Expand the texture if possible
    let shaderLanguage = 0 /* ShaderLanguage.GLSL */;
    if (expandTexture) {
        if (engine.isWebGPU) {
            shaderLanguage = 1 /* ShaderLanguage.WGSL */;
            await __vitePreload(() => import('./rgbdDecode.fragment-DwPrei9n.js'),true              ?__vite__mapDeps([0,1,2,3,4]):void 0,import.meta.url);
        }
        else {
            await __vitePreload(() => import('./rgbdDecode.fragment-PHs4X25h.js'),true              ?__vite__mapDeps([5,1,2,3,6]):void 0,import.meta.url);
        }
        // Simply run through the decode PP
        rgbdPostProcess = new PostProcess("rgbdDecode", "rgbdDecode", null, null, 1, null, 3, engine, false, undefined, texture.type, undefined, null, false, undefined, shaderLanguage);
        texture._isRGBD = false;
        texture.invertY = false;
        cubeRtt = engine.createRenderTargetCubeTexture(texture.width, {
            generateDepthBuffer: false,
            generateMipMaps: true,
            generateStencilBuffer: false,
            samplingMode: 3,
            type: texture.type,
            format: 5,
        });
    }
    else {
        texture._isRGBD = true;
        texture.invertY = true;
        // In case of missing support, applies the same patch than DDS files.
        if (generateNonLODTextures) {
            const mipSlices = 3;
            lodTextures = {};
            const scale = texture._lodGenerationScale;
            const offset = texture._lodGenerationOffset;
            for (let i = 0; i < mipSlices; i++) {
                //compute LOD from even spacing in smoothness (matching shader calculation)
                const smoothness = i / (mipSlices - 1);
                const roughness = 1 - smoothness;
                const minLODIndex = offset; // roughness = 0
                const maxLODIndex = (mipmapsCount - 1) * scale + offset; // roughness = 1 (mipmaps start from 0)
                const lodIndex = minLODIndex + (maxLODIndex - minLODIndex) * roughness;
                const mipmapIndex = Math.round(Math.min(Math.max(lodIndex, 0), maxLODIndex));
                //compute LOD from even spacing in smoothness (matching shader calculation)
                const glTextureFromLod = new InternalTexture(engine, 2 /* InternalTextureSource.Temp */);
                glTextureFromLod.isCube = true;
                glTextureFromLod.invertY = true;
                glTextureFromLod.generateMipMaps = false;
                engine.updateTextureSamplingMode(2, glTextureFromLod);
                // Wrap in a base texture for easy binding.
                const lodTexture = new BaseTexture(null);
                lodTexture._isCube = true;
                lodTexture._texture = glTextureFromLod;
                lodTextures[mipmapIndex] = lodTexture;
                switch (i) {
                    case 0:
                        texture._lodTextureLow = lodTexture;
                        break;
                    case 1:
                        texture._lodTextureMid = lodTexture;
                        break;
                    case 2:
                        texture._lodTextureHigh = lodTexture;
                        break;
                }
            }
        }
    }
    const promises = [];
    // All mipmaps up to provided number of images
    for (let i = 0; i < imageData.length; i++) {
        // All faces
        for (let face = 0; face < 6; face++) {
            // Constructs an image element from image data
            const bytes = imageData[i][face];
            const blob = new Blob([bytes], { type: imageType });
            const url = URL.createObjectURL(blob);
            let promise;
            if (engine._features.forceBitmapOverHTMLImageElement) {
                // eslint-disable-next-line github/no-then
                promise = engine.createImageBitmap(blob, { premultiplyAlpha: "none" }).then(async (img) => {
                    return await _OnImageReadyAsync(img, engine, expandTexture, rgbdPostProcess, url, face, i, generateNonLODTextures, lodTextures, cubeRtt, texture);
                });
            }
            else {
                const image = new Image();
                image.src = url;
                // Enqueue promise to upload to the texture.
                promise = new Promise((resolve, reject) => {
                    image.onload = () => {
                        _OnImageReadyAsync(image, engine, expandTexture, rgbdPostProcess, url, face, i, generateNonLODTextures, lodTextures, cubeRtt, texture)
                            // eslint-disable-next-line github/no-then
                            .then(() => resolve())
                            // eslint-disable-next-line github/no-then
                            .catch((reason) => {
                            // eslint-disable-next-line @typescript-eslint/prefer-promise-reject-errors
                            reject(reason);
                        });
                    };
                    image.onerror = (error) => {
                        // eslint-disable-next-line @typescript-eslint/prefer-promise-reject-errors
                        reject(error);
                    };
                });
            }
            promises.push(promise);
        }
    }
    await Promise.all(promises);
    // Fill remaining mipmaps with black textures.
    if (imageData.length < mipmapsCount) {
        let data;
        const size = Math.pow(2, mipmapsCount - 1 - imageData.length);
        const dataLength = size * size * 4;
        switch (texture.type) {
            case 0: {
                data = new Uint8Array(dataLength);
                break;
            }
            case 2: {
                data = new Uint16Array(dataLength);
                break;
            }
            case 1: {
                data = new Float32Array(dataLength);
                break;
            }
        }
        for (let i = imageData.length; i < mipmapsCount; i++) {
            for (let face = 0; face < 6; face++) {
                engine._uploadArrayBufferViewToTexture(cubeRtt?.texture || texture, data, face, i);
            }
        }
    }
    // Release temp RTT.
    if (cubeRtt) {
        const irradiance = texture._irradianceTexture;
        texture._irradianceTexture = null;
        engine._releaseTexture(texture);
        cubeRtt._swapAndDie(texture);
        texture._irradianceTexture = irradiance;
    }
    // Release temp Post Process.
    if (rgbdPostProcess) {
        rgbdPostProcess.dispose();
    }
    // Flag internal texture as ready in case they are in use.
    if (generateNonLODTextures) {
        if (texture._lodTextureHigh && texture._lodTextureHigh._texture) {
            texture._lodTextureHigh._texture.isReady = true;
        }
        if (texture._lodTextureMid && texture._lodTextureMid._texture) {
            texture._lodTextureMid._texture.isReady = true;
        }
        if (texture._lodTextureLow && texture._lodTextureLow._texture) {
            texture._lodTextureLow._texture.isReady = true;
        }
    }
}
/**
 * Uploads spherical polynomials information to the texture.
 * @param texture defines the texture we are trying to upload the information to
 * @param info defines the environment texture info retrieved through the GetEnvInfo method
 */
function UploadEnvSpherical(texture, info) {
    info = normalizeEnvInfo(info);
    const irradianceInfo = info.irradiance;
    if (!irradianceInfo) {
        return;
    }
    const sp = new SphericalPolynomial();
    Vector3.FromArrayToRef(irradianceInfo.x, 0, sp.x);
    Vector3.FromArrayToRef(irradianceInfo.y, 0, sp.y);
    Vector3.FromArrayToRef(irradianceInfo.z, 0, sp.z);
    Vector3.FromArrayToRef(irradianceInfo.xx, 0, sp.xx);
    Vector3.FromArrayToRef(irradianceInfo.yy, 0, sp.yy);
    Vector3.FromArrayToRef(irradianceInfo.zz, 0, sp.zz);
    Vector3.FromArrayToRef(irradianceInfo.yz, 0, sp.yz);
    Vector3.FromArrayToRef(irradianceInfo.zx, 0, sp.zx);
    Vector3.FromArrayToRef(irradianceInfo.xy, 0, sp.xy);
    texture._sphericalPolynomial = sp;
}
/**
 * @internal
 */
// eslint-disable-next-line @typescript-eslint/promise-function-async, no-restricted-syntax
function _UpdateRGBDAsync(internalTexture, data, sphericalPolynomial, lodScale, lodOffset) {
    const proxy = internalTexture
        .getEngine()
        .createRawCubeTexture(null, internalTexture.width, internalTexture.format, internalTexture.type, internalTexture.generateMipMaps, internalTexture.invertY, internalTexture.samplingMode, internalTexture._compression);
    // eslint-disable-next-line github/no-then
    const proxyPromise = UploadRadianceLevelsAsync(proxy, data).then(() => internalTexture);
    internalTexture.onRebuildCallback = (_internalTexture) => {
        return {
            proxy: proxyPromise,
            isReady: true,
            isAsync: true,
        };
    };
    internalTexture._source = 13 /* InternalTextureSource.CubeRawRGBD */;
    internalTexture._bufferViewArrayArray = data;
    internalTexture._lodGenerationScale = lodScale;
    internalTexture._lodGenerationOffset = lodOffset;
    internalTexture._sphericalPolynomial = sphericalPolynomial;
    // eslint-disable-next-line github/no-then
    return UploadRadianceLevelsAsync(internalTexture, data).then(() => {
        internalTexture.isReady = true;
        return internalTexture;
    });
}

export { GetEnvInfo as G, UploadEnvSpherical as U, _UpdateRGBDAsync as _, UploadEnvLevelsAsync as a };
//# sourceMappingURL=environmentTextureTools-DFwbt9Lr.js.map
