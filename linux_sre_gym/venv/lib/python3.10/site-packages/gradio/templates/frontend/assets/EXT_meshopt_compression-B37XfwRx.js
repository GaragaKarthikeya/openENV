import { GLTFLoader, ArrayItem } from './glTFLoader-BbG05oe0.js';
import { d as Tools, aG as unregisterGLTFExtension, aH as registerGLTFExtension } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';
import './bone-C6xIrJX4.js';
import './skeleton-DzANFt7R.js';
import './rawTexture-Bto4WZO2.js';
import './assetContainer-7GPvH2S_.js';
import './objectModelMapping-DnUo8NMz.js';

// eslint-disable-next-line @typescript-eslint/naming-convention
let NumberOfWorkers = 0;
// eslint-disable-next-line @typescript-eslint/naming-convention
let WorkerTimeout = null;
/**
 * Meshopt compression (https://github.com/zeux/meshoptimizer)
 *
 * This class wraps the meshopt library from https://github.com/zeux/meshoptimizer/tree/master/js.
 *
 * **Encoder**
 *
 * The encoder is not currently implemented.
 *
 * **Decoder**
 *
 * By default, the configuration points to a copy of the meshopt files on the Babylon.js preview CDN (e.g. https://preview.babylonjs.com/meshopt_decoder.js).
 *
 * To update the configuration, use the following code:
 * ```javascript
 *     MeshoptCompression.Configuration = {
 *         decoder: {
 *             url: "<url to the meshopt decoder library>"
 *         }
 *     };
 * ```
 */
class MeshoptCompression {
    /**
     * Default instance for the meshoptimizer object.
     */
    static get Default() {
        if (!MeshoptCompression._Default) {
            MeshoptCompression._Default = new MeshoptCompression();
        }
        return MeshoptCompression._Default;
    }
    /**
     * Constructor
     */
    constructor() {
        const decoder = MeshoptCompression.Configuration.decoder;
        // eslint-disable-next-line github/no-then
        this._decoderModulePromise = Tools.LoadBabylonScriptAsync(decoder.url).then(() => {
            // Wait for WebAssembly compilation before resolving promise
            return MeshoptDecoder.ready;
        });
    }
    /**
     * Stop all async operations and release resources.
     */
    dispose() {
        delete this._decoderModulePromise;
    }
    /**
     * Decode meshopt data.
     * @see https://github.com/zeux/meshoptimizer/tree/master/js#decoder
     * @param source The input data.
     * @param count The number of elements.
     * @param stride The stride in bytes.
     * @param mode The compression mode.
     * @param filter The compression filter.
     * @returns a Promise<Uint8Array> that resolves to the decoded data
     */
    async decodeGltfBufferAsync(source, count, stride, mode, filter) {
        await this._decoderModulePromise;
        if (NumberOfWorkers === 0) {
            MeshoptDecoder.useWorkers(1);
            NumberOfWorkers = 1;
        }
        const result = await MeshoptDecoder.decodeGltfBufferAsync(count, stride, source, mode, filter);
        // a simple debounce to avoid switching back and forth between workers and no workers while decoding
        if (WorkerTimeout !== null) {
            clearTimeout(WorkerTimeout);
        }
        WorkerTimeout = setTimeout(() => {
            MeshoptDecoder.useWorkers(0);
            NumberOfWorkers = 0;
            WorkerTimeout = null;
        }, 1000);
        return result;
    }
}
/**
 * The configuration. Defaults to the following:
 * ```javascript
 * decoder: {
 *   url: "https://cdn.babylonjs.com/meshopt_decoder.js"
 * }
 * ```
 */
MeshoptCompression.Configuration = {
    decoder: {
        url: `${Tools._DefaultCdnUrl}/meshopt_decoder.js`,
    },
};
MeshoptCompression._Default = null;

const NAME = "EXT_meshopt_compression";
/**
 * [Specification](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Vendor/EXT_meshopt_compression/README.md)
 *
 * This extension uses a WebAssembly decoder module from https://github.com/zeux/meshoptimizer/tree/master/js
 * @since 5.0.0
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
class EXT_meshopt_compression {
    /**
     * @internal
     */
    constructor(loader) {
        /**
         * The name of this extension.
         */
        this.name = NAME;
        this.enabled = loader.isExtensionUsed(NAME);
        this._loader = loader;
    }
    /** @internal */
    dispose() {
        this._loader = null;
    }
    /**
     * @internal
     */
    // eslint-disable-next-line no-restricted-syntax
    loadBufferViewAsync(context, bufferView) {
        return GLTFLoader.LoadExtensionAsync(context, bufferView, this.name, async (extensionContext, extension) => {
            const bufferViewMeshopt = bufferView;
            if (bufferViewMeshopt._meshOptData) {
                return await bufferViewMeshopt._meshOptData;
            }
            const buffer = ArrayItem.Get(`${context}/buffer`, this._loader.gltf.buffers, extension.buffer);
            bufferViewMeshopt._meshOptData = this._loader
                .loadBufferAsync(`/buffers/${buffer.index}`, buffer, extension.byteOffset || 0, extension.byteLength)
                // eslint-disable-next-line github/no-then
                .then(async (buffer) => {
                return await MeshoptCompression.Default.decodeGltfBufferAsync(buffer, extension.count, extension.byteStride, extension.mode, extension.filter);
            });
            return await bufferViewMeshopt._meshOptData;
        });
    }
}
unregisterGLTFExtension(NAME);
registerGLTFExtension(NAME, true, (loader) => new EXT_meshopt_compression(loader));

export { EXT_meshopt_compression };
//# sourceMappingURL=EXT_meshopt_compression-B37XfwRx.js.map
