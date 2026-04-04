import { O as Observable, V as Vector3, aq as EngineStore, A as AbstractEngine, L as Logger, bo as _RetryWithInterval, be as _WarnImport, R as RegisterClass, b as Scene, a as SceneComponentConstants, M as Matrix, by as PrecisionDate, d as Tools, aG as unregisterGLTFExtension, aH as registerGLTFExtension } from './index-CxR0Yf_7.js';
import { _ as _HasSpatialAudioOptions, d as _SpatialAudioDefaults } from './spatialWebAudio-DH890j7t.js';
import { _WebAudioSoundSource } from './webAudioSoundSource-__1CMM9A.js';
import { _WebAudioStaticSound } from './webAudioStaticSound-D8ZwU_iO.js';
import { _WebAudioStreamingSound } from './webAudioStreamingSound-CvXX6uFt.js';
import { ArrayItem, GLTFLoader } from './glTFLoader-BbG05oe0.js';
import './audioEngine-CXlZUK-6.js';
import './index-BvBk1Iap.js';
import './webAudioBaseSubGraph-CuchFrSM.js';
import './abstractSoundSource-Bo3P1qg1.js';
import './abstractSoundInstance-CH6fBm2E.js';
import './bone-C6xIrJX4.js';
import './skeleton-DzANFt7R.js';
import './rawTexture-Bto4WZO2.js';
import './assetContainer-7GPvH2S_.js';
import './objectModelMapping-DnUo8NMz.js';

/**
 * Composed of a frame, and an action function
 */
class AnimationEvent {
    /**
     * Initializes the animation event
     * @param frame The frame for which the event is triggered
     * @param action The event to perform when triggered
     * @param onlyOnce Specifies if the event should be triggered only once
     */
    constructor(
    /** The frame for which the event is triggered **/
    frame, 
    /** The event to perform when triggered **/
    action, 
    /** Specifies if the event should be triggered only once**/
    onlyOnce) {
        this.frame = frame;
        this.action = action;
        this.onlyOnce = onlyOnce;
        /**
         * Specifies if the animation event is done
         */
        this.isDone = false;
    }
    /** @internal */
    _clone() {
        return new AnimationEvent(this.frame, this.action, this.onlyOnce);
    }
}

const TmpRampOptions = {
    duration: 0,
    shape: "linear" /* AudioParameterRampShape.Linear */,
};
const TmpPlayOptions = {
    duration: 0,
    startOffset: 0,
    waitTime: 0,
};
const TmpStopOptions = {
    waitTime: 0,
};
function D2r(degrees) {
    return (degrees * Math.PI) / 180;
}
function R2d(radians) {
    return (radians * 180) / Math.PI;
}
/**
 * Defines a sound that can be played in the application.
 * The sound can either be an ambient track or a simple sound played in reaction to a user action.
 * @see https://doc.babylonjs.com/legacy/audio
 */
class Sound {
    /**
     * The name of the sound in the scene.
     */
    get name() {
        return this._soundV2.name;
    }
    set name(value) {
        this._soundV2.name = value;
    }
    /**
     * Does the sound autoplay once loaded.
     */
    get autoplay() {
        return this._soundV2 instanceof _WebAudioSoundSource ? true : this._optionsV2.autoplay;
    }
    set autoplay(value) {
        this._optionsV2.autoplay = value;
    }
    /**
     * Does the sound loop after it finishes playing once.
     */
    get loop() {
        return this._soundV2 instanceof _WebAudioSoundSource ? true : this._soundV2.loop;
    }
    set loop(value) {
        if (this._soundV2 instanceof _WebAudioSoundSource) {
            return;
        }
        if (this._soundV2) {
            this._soundV2.loop = value;
        }
    }
    /**
     * Is this sound currently played.
     */
    get isPlaying() {
        return this._soundV2 instanceof _WebAudioSoundSource ? true : this._soundV2?.state === 3 /* SoundState.Started */ || (!this.isReady() && this._optionsV2.autoplay);
    }
    /**
     * Is this sound currently paused.
     */
    get isPaused() {
        return this._soundV2 instanceof _WebAudioSoundSource ? false : this._soundV2.state === 5 /* SoundState.Paused */;
    }
    /**
     * Define the max distance the sound should be heard (intensity just became 0 at this point).
     * @see https://doc.babylonjs.com/legacy/audio#creating-a-spatial-3d-sound
     */
    get maxDistance() {
        return this._optionsV2.spatialMaxDistance || 100;
    }
    set maxDistance(value) {
        this._optionsV2.spatialMaxDistance = value;
        if (this.useCustomAttenuation) {
            return;
        }
        if (this._soundV2) {
            this._initSpatial();
            this._soundV2.spatial.maxDistance = value;
        }
    }
    /**
     * Define the distance attenuation model the sound will follow.
     * @see https://doc.babylonjs.com/legacy/audio#creating-a-spatial-3d-sound
     */
    get distanceModel() {
        return this._optionsV2.spatialDistanceModel || "linear";
    }
    set distanceModel(value) {
        this._optionsV2.spatialDistanceModel = value;
        if (this._soundV2) {
            this._initSpatial();
            this._soundV2.spatial.distanceModel = value;
        }
    }
    /**
     * Gets the current time for the sound.
     */
    get currentTime() {
        return this._soundV2 instanceof _WebAudioSoundSource ? this._soundV2.engine.currentTime : this._soundV2.currentTime;
    }
    /**
     * Does this sound enables spatial sound.
     * @see https://doc.babylonjs.com/legacy/audio#creating-a-spatial-3d-sound
     */
    get spatialSound() {
        return this._soundV2?._isSpatial ?? false;
    }
    /**
     * Does this sound enables spatial sound.
     * @see https://doc.babylonjs.com/legacy/audio#creating-a-spatial-3d-sound
     */
    set spatialSound(newValue) {
        if (this._soundV2) {
            if (newValue) {
                this._initSpatial();
            }
            else {
                this._soundV2._isSpatial = false;
            }
        }
    }
    get _onReady() {
        if (!this._onReadyObservable) {
            this._onReadyObservable = new Observable();
        }
        return this._onReadyObservable;
    }
    /**
     * Create a sound and attach it to a scene
     * @param name Name of your sound
     * @param urlOrArrayBuffer Url to the sound to load async or ArrayBuffer, it also works with MediaStreams and AudioBuffers
     * @param scene defines the scene the sound belongs to
     * @param readyToPlayCallback Provide a callback function if you'd like to load your code once the sound is ready to be played
     * @param options Objects to provide with the current available options: autoplay, loop, volume, spatialSound, maxDistance, rolloffFactor, refDistance, distanceModel, panningModel, streaming
     */
    constructor(name, urlOrArrayBuffer, scene, readyToPlayCallback = null, options) {
        /**
         * Does the sound use a custom attenuation curve to simulate the falloff
         * happening when the source gets further away from the camera.
         * @see https://doc.babylonjs.com/legacy/audio#creating-your-own-custom-attenuation-function
         */
        this.useCustomAttenuation = false;
        /**
         * The sound track id this sound belongs to.
         */
        this.soundTrackId = -1;
        /**
         * Define the reference distance the sound should be heard perfectly.
         * @see https://doc.babylonjs.com/legacy/audio#creating-a-spatial-3d-sound
         */
        this.refDistance = 1;
        /**
         * Define the roll off factor of spatial sounds.
         * @see https://doc.babylonjs.com/legacy/audio#creating-a-spatial-3d-sound
         */
        this.rolloffFactor = 1;
        /**
         * Gets or sets an object used to store user defined information for the sound.
         */
        this.metadata = null;
        /**
         * Observable event when the current playing sound finishes.
         */
        this.onEndedObservable = new Observable();
        this._localDirection = new Vector3(1, 0, 0);
        this._volume = 1;
        this._isReadyToPlay = false;
        this._isDirectional = false;
        this._isOutputConnected = false;
        this._url = null;
        this._onReadyObservable = null;
        this._onReadyToPlay = () => {
            this._scene.mainSoundTrack.addSound(this);
            this._isReadyToPlay = true;
            this._readyToPlayCallback();
            if (this._onReadyObservable) {
                this._onReadyObservable.notifyObservers();
            }
            if (this._optionsV2.autoplay) {
                this.play();
            }
        };
        this._onended = () => {
            if (this.onended) {
                this.onended();
            }
            this.onEndedObservable.notifyObservers(this);
        };
        scene = scene || EngineStore.LastCreatedScene;
        if (!scene) {
            return;
        }
        this._scene = scene;
        Sound._SceneComponentInitialization(scene);
        this._readyToPlayCallback = readyToPlayCallback || (() => { });
        // Default custom attenuation function is a linear attenuation
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        this._customAttenuationFunction = (currentVolume, currentDistance, maxDistance, refDistance, rolloffFactor) => {
            if (currentDistance < maxDistance) {
                return currentVolume * (1 - currentDistance / maxDistance);
            }
            else {
                return 0;
            }
        };
        options = options || {};
        const optionsV2 = {
            analyzerEnabled: false,
            autoplay: false, // `false` for now, but will be set to given option later
            duration: options.length || 0,
            loop: options.loop || false,
            loopEnd: 0,
            loopStart: 0,
            outBus: null,
            outBusAutoDefault: false,
            playbackRate: options.playbackRate || 1,
            pitch: 0,
            skipCodecCheck: options.skipCodecCheck || false,
            spatialDistanceModel: options.distanceModel,
            spatialEnabled: options.spatialSound,
            spatialMaxDistance: options.maxDistance,
            spatialMinDistance: options.refDistance,
            spatialRolloffFactor: options.rolloffFactor,
            stereoEnabled: false,
            startOffset: options.offset || 0,
            volume: options.volume ?? 1,
        };
        this._volume = options.volume ?? 1;
        if (_HasSpatialAudioOptions(optionsV2)) {
            optionsV2.spatialAutoUpdate = false;
            optionsV2.spatialConeInnerAngle = _SpatialAudioDefaults.coneInnerAngle;
            optionsV2.spatialConeOuterAngle = _SpatialAudioDefaults.coneOuterAngle;
            optionsV2.spatialConeOuterVolume = _SpatialAudioDefaults.coneOuterVolume;
            optionsV2.spatialMinUpdateTime = 0;
            optionsV2.spatialOrientation = _SpatialAudioDefaults.orientation.clone();
            optionsV2.spatialPanningModel = (this._scene.headphone ? "HRTF" : "equalpower");
            optionsV2.spatialPosition = _SpatialAudioDefaults.position.clone();
            optionsV2.spatialRotation = _SpatialAudioDefaults.rotation.clone();
            optionsV2.spatialRotationQuaternion = _SpatialAudioDefaults.rotationQuaternion.clone();
            if (optionsV2.spatialMaxDistance === undefined) {
                optionsV2.spatialMaxDistance = 100;
            }
        }
        this._optionsV2 = { ...optionsV2 };
        this._optionsV2.autoplay = options.autoplay || false;
        this.useCustomAttenuation = options.useCustomAttenuation ?? false;
        if (this.useCustomAttenuation) {
            optionsV2.spatialMaxDistance = Number.MAX_VALUE;
            optionsV2.volume = 0;
        }
        let streaming = options?.streaming || false;
        const audioEngine = AbstractEngine.audioEngine;
        if (!audioEngine) {
            return;
        }
        const audioEngineV2 = AbstractEngine.audioEngine._v2;
        const createSoundV2 = () => {
            if (streaming) {
                const streamingOptionsV2 = {
                    preloadCount: 0,
                    ...optionsV2,
                };
                const sound = new _WebAudioStreamingSound(name, audioEngineV2, streamingOptionsV2);
                // eslint-disable-next-line github/no-then
                void sound._initAsync(urlOrArrayBuffer, optionsV2).then(() => {
                    // eslint-disable-next-line github/no-then
                    void sound.preloadInstancesAsync(1).then(this._onReadyToPlay);
                });
                return sound;
            }
            else {
                const sound = new _WebAudioStaticSound(name, audioEngineV2, optionsV2);
                // eslint-disable-next-line github/no-then
                void sound._initAsync(urlOrArrayBuffer, optionsV2).then(this._onReadyToPlay);
                return sound;
            }
        };
        // If no parameter is passed then the setAudioBuffer should be called to prepare the sound.
        if (!urlOrArrayBuffer) {
            // Create the sound but don't call _initAsync on it, yet. Call it later when `setAudioBuffer` is called.
            this._soundV2 = new _WebAudioStaticSound(name, audioEngineV2, optionsV2);
        }
        else if (typeof urlOrArrayBuffer === "string") {
            this._url = urlOrArrayBuffer;
            this._soundV2 = createSoundV2();
        }
        else if (urlOrArrayBuffer instanceof ArrayBuffer) {
            streaming = false;
            this._soundV2 = createSoundV2();
        }
        else if (urlOrArrayBuffer instanceof HTMLMediaElement) {
            streaming = true;
            this._soundV2 = createSoundV2();
        }
        else if (urlOrArrayBuffer instanceof MediaStream) {
            const node = new MediaStreamAudioSourceNode(audioEngineV2._audioContext, { mediaStream: urlOrArrayBuffer });
            this._soundV2 = new _WebAudioSoundSource(name, node, audioEngineV2, optionsV2);
            // eslint-disable-next-line github/no-then
            void this._soundV2._initAsync(optionsV2).then(this._onReadyToPlay);
        }
        else if (urlOrArrayBuffer instanceof AudioBuffer) {
            streaming = false;
            this._soundV2 = createSoundV2();
        }
        else if (Array.isArray(urlOrArrayBuffer)) {
            this._soundV2 = createSoundV2();
        }
        if (!this._soundV2) {
            Logger.Error("Parameter must be a URL to the sound, an Array of URLs (.mp3 & .ogg) or an ArrayBuffer of the sound.");
            return;
        }
        if (!(this._soundV2 instanceof _WebAudioSoundSource)) {
            this._soundV2.onEndedObservable.add(this._onended);
        }
    }
    /**
     * Release the sound and its associated resources
     */
    dispose() {
        if (this.isPlaying) {
            this.stop();
        }
        this._isReadyToPlay = false;
        if (this.soundTrackId === -1) {
            this._scene.mainSoundTrack.removeSound(this);
        }
        else if (this._scene.soundTracks) {
            this._scene.soundTracks[this.soundTrackId].removeSound(this);
        }
        if (this._connectedTransformNode && this._registerFunc) {
            this._connectedTransformNode.unregisterAfterWorldMatrixUpdate(this._registerFunc);
            this._connectedTransformNode = null;
        }
        this._soundV2.dispose();
    }
    /**
     * Gets if the sounds is ready to be played or not.
     * @returns true if ready, otherwise false
     */
    isReady() {
        return this._isReadyToPlay;
    }
    /**
     * Get the current class name.
     * @returns current class name
     */
    getClassName() {
        return "Sound";
    }
    /**
     * Sets the data of the sound from an audiobuffer
     * @param audioBuffer The audioBuffer containing the data
     */
    setAudioBuffer(audioBuffer) {
        if (this._isReadyToPlay) {
            return;
        }
        if (this._soundV2 instanceof _WebAudioStaticSound) {
            // eslint-disable-next-line @typescript-eslint/no-floating-promises, github/no-then
            this._soundV2._initAsync(audioBuffer, this._optionsV2).then(this._onReadyToPlay);
        }
    }
    /**
     * Updates the current sounds options such as maxdistance, loop...
     * @param options A JSON object containing values named as the object properties
     */
    updateOptions(options) {
        if (options) {
            this.loop = options.loop ?? this.loop;
            this.maxDistance = options.maxDistance ?? this.maxDistance;
            this.useCustomAttenuation = options.useCustomAttenuation ?? this.useCustomAttenuation;
            this.rolloffFactor = options.rolloffFactor ?? this.rolloffFactor;
            this.refDistance = options.refDistance ?? this.refDistance;
            this.distanceModel = options.distanceModel ?? this.distanceModel;
            if (options.playbackRate !== undefined) {
                this.setPlaybackRate(options.playbackRate);
            }
            if (options.spatialSound !== undefined) {
                this.spatialSound = options.spatialSound;
            }
            if (options.volume !== undefined) {
                this.setVolume(options.volume);
            }
            if (this._soundV2 instanceof _WebAudioStaticSound) {
                let updated = false;
                if (options.offset !== undefined) {
                    this._optionsV2.startOffset = options.offset;
                    updated = true;
                }
                if (options.length !== undefined) {
                    this._soundV2.duration = options.length;
                    updated = true;
                }
                if (updated && this.isPaused) {
                    this.stop();
                }
            }
            this._updateSpatialParameters();
        }
    }
    _updateSpatialParameters() {
        if (!this.spatialSound) {
            return;
        }
        const spatial = this._soundV2.spatial;
        if (this.useCustomAttenuation) {
            // Disable WebAudio attenuation.
            spatial.distanceModel = "linear";
            spatial.minDistance = 1;
            spatial.maxDistance = Number.MAX_VALUE;
            spatial.rolloffFactor = 1;
            spatial.panningModel = "equalpower";
        }
        else {
            spatial.distanceModel = this.distanceModel;
            spatial.minDistance = this.refDistance;
            spatial.maxDistance = this.maxDistance;
            spatial.rolloffFactor = this.rolloffFactor;
            spatial.panningModel = this._optionsV2.spatialPanningModel || "equalpower";
        }
    }
    /**
     * Switch the panning model to HRTF:
     * Renders a stereo output of higher quality than equalpower — it uses a convolution with measured impulse responses from human subjects.
     * @see https://doc.babylonjs.com/legacy/audio#creating-a-spatial-3d-sound
     */
    switchPanningModelToHRTF() {
        if (this.spatialSound) {
            this._initSpatial();
            this._soundV2.spatial.panningModel = "HRTF";
        }
    }
    /**
     * Switch the panning model to Equal Power:
     * Represents the equal-power panning algorithm, generally regarded as simple and efficient. equalpower is the default value.
     * @see https://doc.babylonjs.com/legacy/audio#creating-a-spatial-3d-sound
     */
    switchPanningModelToEqualPower() {
        if (this.spatialSound) {
            this._initSpatial();
            this._soundV2.spatial.panningModel = "equalpower";
        }
    }
    /**
     * Connect this sound to a sound track audio node like gain...
     * @param soundTrackAudioNode the sound track audio node to connect to
     */
    connectToSoundTrackAudioNode(soundTrackAudioNode) {
        const outputNode = this._soundV2._outNode;
        if (outputNode) {
            if (this._isOutputConnected) {
                outputNode.disconnect();
            }
            outputNode.connect(soundTrackAudioNode);
            this._isOutputConnected = true;
        }
    }
    /**
     * Transform this sound into a directional source
     * @param coneInnerAngle Size of the inner cone in degree
     * @param coneOuterAngle Size of the outer cone in degree
     * @param coneOuterGain Volume of the sound outside the outer cone (between 0.0 and 1.0)
     */
    setDirectionalCone(coneInnerAngle, coneOuterAngle, coneOuterGain) {
        if (coneOuterAngle < coneInnerAngle) {
            Logger.Error("setDirectionalCone(): outer angle of the cone must be superior or equal to the inner angle.");
            return;
        }
        this._optionsV2.spatialConeInnerAngle = D2r(coneInnerAngle);
        this._optionsV2.spatialConeOuterAngle = D2r(coneOuterAngle);
        this._optionsV2.spatialConeOuterVolume = coneOuterGain;
        this._initSpatial();
        this._soundV2.spatial.coneInnerAngle = this._optionsV2.spatialConeInnerAngle;
        this._soundV2.spatial.coneOuterAngle = this._optionsV2.spatialConeOuterAngle;
        this._soundV2.spatial.coneOuterVolume = coneOuterGain;
        this._isDirectional = true;
        if (this.isPlaying && this.loop) {
            this.stop();
            this.play(0, this._optionsV2.startOffset, this._optionsV2.duration);
        }
    }
    /**
     * Gets or sets the inner angle for the directional cone.
     */
    get directionalConeInnerAngle() {
        return R2d(typeof this._optionsV2.spatialConeInnerAngle === "number" ? this._optionsV2.spatialConeInnerAngle : _SpatialAudioDefaults.coneInnerAngle);
    }
    /**
     * Gets or sets the inner angle for the directional cone.
     */
    set directionalConeInnerAngle(value) {
        value = D2r(value);
        if (value != this._optionsV2.spatialConeInnerAngle) {
            if (this.directionalConeOuterAngle < value) {
                Logger.Error("directionalConeInnerAngle: outer angle of the cone must be superior or equal to the inner angle.");
                return;
            }
            this._optionsV2.spatialConeInnerAngle = value;
            if (this.spatialSound) {
                this._initSpatial();
                this._soundV2.spatial.coneInnerAngle = value;
            }
        }
    }
    /**
     * Gets or sets the outer angle for the directional cone.
     */
    get directionalConeOuterAngle() {
        return R2d(typeof this._optionsV2.spatialConeOuterAngle === "number" ? this._optionsV2.spatialConeOuterAngle : _SpatialAudioDefaults.coneOuterAngle);
    }
    /**
     * Gets or sets the outer angle for the directional cone.
     */
    set directionalConeOuterAngle(value) {
        value = D2r(value);
        if (value != this._optionsV2.spatialConeOuterAngle) {
            if (value < this.directionalConeInnerAngle) {
                Logger.Error("directionalConeOuterAngle: outer angle of the cone must be superior or equal to the inner angle.");
                return;
            }
            this._optionsV2.spatialConeOuterAngle = value;
            if (this.spatialSound) {
                this._initSpatial();
                this._soundV2.spatial.coneOuterAngle = value;
            }
        }
    }
    /**
     * Sets the position of the emitter if spatial sound is enabled
     * @param newPosition Defines the new position
     */
    setPosition(newPosition) {
        if (this._optionsV2.spatialPosition && newPosition.equals(this._optionsV2.spatialPosition)) {
            return;
        }
        if (!this._optionsV2.spatialPosition) {
            this._optionsV2.spatialPosition = Vector3.Zero();
        }
        this._optionsV2.spatialPosition.copyFrom(newPosition);
        if (this.spatialSound && !isNaN(newPosition.x) && !isNaN(newPosition.y) && !isNaN(newPosition.z)) {
            this._initSpatial();
            this._soundV2.spatial.position = newPosition;
        }
    }
    /**
     * Sets the local direction of the emitter if spatial sound is enabled
     * @param newLocalDirection Defines the new local direction
     */
    setLocalDirectionToMesh(newLocalDirection) {
        this._localDirection = newLocalDirection;
        if (this._connectedTransformNode && this.isPlaying) {
            this._updateDirection();
        }
    }
    _updateDirection() {
        if (!this._connectedTransformNode || !this.spatialSound) {
            return;
        }
        const mat = this._connectedTransformNode.getWorldMatrix();
        const direction = Vector3.TransformNormal(this._localDirection, mat);
        direction.normalize();
        this._initSpatial();
        this._soundV2.spatial.orientation = direction;
    }
    _initSpatial() {
        this._soundV2._isSpatial = true;
        if (this._optionsV2.spatialDistanceModel === undefined) {
            this._optionsV2.spatialDistanceModel = "linear";
            this._soundV2.spatial.distanceModel = "linear";
        }
        if (this._optionsV2.spatialMaxDistance === undefined) {
            this._optionsV2.spatialMaxDistance = 100;
            this._soundV2.spatial.maxDistance = 100;
        }
    }
    /** @internal */
    updateDistanceFromListener() {
        if (this._soundV2._outNode && this._connectedTransformNode && this.useCustomAttenuation && this._scene.activeCamera) {
            const distance = this._scene.audioListenerPositionProvider
                ? this._connectedTransformNode.position.subtract(this._scene.audioListenerPositionProvider()).length()
                : this._connectedTransformNode.getDistanceToCamera(this._scene.activeCamera);
            this._soundV2.volume = this._customAttenuationFunction(this._volume, distance, this.maxDistance, this.refDistance, this.rolloffFactor);
        }
    }
    /**
     * Sets a new custom attenuation function for the sound.
     * @param callback Defines the function used for the attenuation
     * @see https://doc.babylonjs.com/legacy/audio#creating-your-own-custom-attenuation-function
     */
    setAttenuationFunction(callback) {
        this._customAttenuationFunction = callback;
    }
    /**
     * Play the sound
     * @param time (optional) Start the sound after X seconds. Start immediately (0) by default.
     * @param offset (optional) Start the sound at a specific time in seconds
     * @param length (optional) Sound duration (in seconds)
     */
    play(time, offset, length) {
        const audioEngine = AbstractEngine.audioEngine;
        audioEngine?.unlock();
        // WebAudio sound sources have no `play` function because they are always playing.
        if (this._soundV2 instanceof _WebAudioSoundSource) {
            return;
        }
        if (this._isReadyToPlay && this._scene.audioEnabled) {
            // The sound can only resume from pause when the `time`, `offset` and `length` args are not set.
            if (this._soundV2.state === 5 /* SoundState.Paused */ && (time !== undefined || offset !== undefined || length !== undefined)) {
                this._soundV2.stop();
            }
            try {
                TmpPlayOptions.duration = length || 0;
                TmpPlayOptions.startOffset = offset !== undefined ? offset || this._optionsV2.startOffset : this._optionsV2.startOffset;
                TmpPlayOptions.waitTime = time || 0;
                if (audioEngine?.unlocked) {
                    this._soundV2.play(TmpPlayOptions);
                }
                else {
                    // Wait a bit for FF as context seems late to be ready.
                    setTimeout(() => {
                        this._soundV2.play(TmpPlayOptions);
                    }, 500);
                }
            }
            catch (ex) {
                Logger.Error("Error while trying to play audio: " + this.name + ", " + ex.message);
            }
        }
    }
    /**
     * Stop the sound
     * @param time (optional) Stop the sound after X seconds. Stop immediately (0) by default.
     */
    stop(time) {
        if (!this._soundV2) {
            return;
        }
        // WebAudio sound sources have no `stop` function because they are always playing.
        if (this._soundV2 instanceof _WebAudioSoundSource) {
            return;
        }
        TmpStopOptions.waitTime = time || 0;
        this._soundV2.stop(TmpStopOptions);
    }
    /**
     * Put the sound in pause
     */
    pause() {
        if (!this._soundV2) {
            return;
        }
        // WebAudio sound sources have no `pause` function because they are always playing.
        if (this._soundV2 instanceof _WebAudioSoundSource) {
            return;
        }
        this._soundV2.pause();
    }
    /**
     * Sets a dedicated volume for this sounds
     * @param newVolume Define the new volume of the sound
     * @param time Define time for gradual change to new volume
     */
    setVolume(newVolume, time) {
        if (!this.isReady()) {
            this._onReady.addOnce(() => {
                this.setVolume(newVolume, time);
            });
            return;
        }
        TmpRampOptions.duration = time || 0;
        this._soundV2.setVolume(newVolume, TmpRampOptions);
        this._volume = newVolume;
    }
    /**
     * Set the sound play back rate
     * @param newPlaybackRate Define the playback rate the sound should be played at
     */
    setPlaybackRate(newPlaybackRate) {
        if (this._soundV2 instanceof _WebAudioStaticSound) {
            this._soundV2.playbackRate = newPlaybackRate;
        }
    }
    /**
     * Gets the sound play back rate.
     * @returns the  play back rate of the sound
     */
    getPlaybackRate() {
        if (this._soundV2 instanceof _WebAudioStaticSound) {
            return this._soundV2.playbackRate;
        }
        return 1;
    }
    /**
     * Gets the volume of the sound.
     * @returns the volume of the sound
     */
    getVolume() {
        return this._volume;
    }
    /**
     * Attach the sound to a dedicated mesh
     * @param transformNode The transform node to connect the sound with
     * @see https://doc.babylonjs.com/legacy/audio#attaching-a-sound-to-a-mesh
     */
    attachToMesh(transformNode) {
        if (this._connectedTransformNode && this._registerFunc) {
            this._connectedTransformNode.unregisterAfterWorldMatrixUpdate(this._registerFunc);
            this._registerFunc = null;
        }
        this._connectedTransformNode = transformNode;
        if (!this.spatialSound) {
            this.spatialSound = true;
            if (this.isPlaying && this.loop) {
                this.stop();
                this.play(0, this._optionsV2.startOffset, this._optionsV2.duration);
            }
        }
        this._onRegisterAfterWorldMatrixUpdate(this._connectedTransformNode);
        this._registerFunc = (transformNode) => this._onRegisterAfterWorldMatrixUpdate(transformNode);
        this._connectedTransformNode.registerAfterWorldMatrixUpdate(this._registerFunc);
    }
    /**
     * Detach the sound from the previously attached mesh
     * @see https://doc.babylonjs.com/legacy/audio#attaching-a-sound-to-a-mesh
     */
    detachFromMesh() {
        if (this._connectedTransformNode && this._registerFunc) {
            this._connectedTransformNode.unregisterAfterWorldMatrixUpdate(this._registerFunc);
            this._registerFunc = null;
            this._connectedTransformNode = null;
        }
    }
    _onRegisterAfterWorldMatrixUpdate(node) {
        if (!node.getBoundingInfo) {
            this.setPosition(node.absolutePosition);
        }
        else {
            const mesh = node;
            const boundingInfo = mesh.getBoundingInfo();
            this.setPosition(boundingInfo.boundingSphere.centerWorld);
        }
        if (this._isDirectional && this.isPlaying) {
            this._updateDirection();
        }
    }
    /**
     * Clone the current sound in the scene.
     * @returns the new sound clone
     */
    clone() {
        if (!(this._soundV2 instanceof _WebAudioStaticSound)) {
            return null;
        }
        const currentOptions = {
            autoplay: this.autoplay,
            loop: this.loop,
            volume: this._volume,
            spatialSound: this.spatialSound,
            maxDistance: this.maxDistance,
            useCustomAttenuation: this.useCustomAttenuation,
            rolloffFactor: this.rolloffFactor,
            refDistance: this.refDistance,
            distanceModel: this.distanceModel,
        };
        const clonedSound = new Sound(this.name + "_cloned", this._soundV2.buffer, this._scene, null, currentOptions);
        clonedSound._optionsV2 = this._optionsV2;
        if (this.useCustomAttenuation) {
            clonedSound.setAttenuationFunction(this._customAttenuationFunction);
        }
        return clonedSound;
    }
    /**
     * Gets the current underlying audio buffer containing the data
     * @returns the audio buffer
     */
    getAudioBuffer() {
        if (this._soundV2 instanceof _WebAudioStaticSound) {
            return this._soundV2.buffer._audioBuffer;
        }
        return null;
    }
    /**
     * Gets the WebAudio AudioBufferSourceNode, lets you keep track of and stop instances of this Sound.
     * @returns the source node
     */
    getSoundSource() {
        // return this._soundSource;
        return null;
    }
    /**
     * Gets the WebAudio GainNode, gives you precise control over the gain of instances of this Sound.
     * @returns the gain node
     */
    getSoundGain() {
        return this._soundV2._outNode;
    }
    /**
     * Serializes the Sound in a JSON representation
     * @returns the JSON representation of the sound
     */
    serialize() {
        const serializationObject = {
            name: this.name,
            url: this._url,
            autoplay: this.autoplay,
            loop: this.loop,
            volume: this._volume,
            spatialSound: this.spatialSound,
            maxDistance: this.maxDistance,
            rolloffFactor: this.rolloffFactor,
            refDistance: this.refDistance,
            distanceModel: this.distanceModel,
            playbackRate: this.getPlaybackRate(),
            panningModel: this._soundV2.spatial.panningModel,
            soundTrackId: this.soundTrackId,
            metadata: this.metadata,
        };
        if (this.spatialSound) {
            if (this._connectedTransformNode) {
                serializationObject.connectedMeshId = this._connectedTransformNode.id;
            }
            serializationObject.position = this._soundV2.spatial.position.asArray();
            serializationObject.refDistance = this.refDistance;
            serializationObject.distanceModel = this.distanceModel;
            serializationObject.isDirectional = this._isDirectional;
            serializationObject.localDirectionToMesh = this._localDirection.asArray();
            serializationObject.coneInnerAngle = this.directionalConeInnerAngle;
            serializationObject.coneOuterAngle = this.directionalConeOuterAngle;
            serializationObject.coneOuterGain = this._soundV2.spatial.coneOuterVolume;
        }
        return serializationObject;
    }
    /**
     * Parse a JSON representation of a sound to instantiate in a given scene
     * @param parsedSound Define the JSON representation of the sound (usually coming from the serialize method)
     * @param scene Define the scene the new parsed sound should be created in
     * @param rootUrl Define the rooturl of the load in case we need to fetch relative dependencies
     * @param sourceSound Define a sound place holder if do not need to instantiate a new one
     * @returns the newly parsed sound
     */
    static Parse(parsedSound, scene, rootUrl, sourceSound) {
        const soundName = parsedSound.name;
        let soundUrl;
        if (parsedSound.url) {
            soundUrl = rootUrl + parsedSound.url;
        }
        else {
            soundUrl = rootUrl + soundName;
        }
        const options = {
            autoplay: parsedSound.autoplay,
            loop: parsedSound.loop,
            volume: parsedSound.volume,
            spatialSound: parsedSound.spatialSound,
            maxDistance: parsedSound.maxDistance,
            rolloffFactor: parsedSound.rolloffFactor,
            refDistance: parsedSound.refDistance,
            distanceModel: parsedSound.distanceModel,
            playbackRate: parsedSound.playbackRate,
        };
        let newSound;
        if (!sourceSound) {
            newSound = new Sound(soundName, soundUrl, scene, () => {
                scene.removePendingData(newSound);
            }, options);
            scene.addPendingData(newSound);
        }
        else {
            const setBufferAndRun = () => {
                _RetryWithInterval(() => sourceSound._isReadyToPlay, () => {
                    const audioBuffer = sourceSound.getAudioBuffer();
                    if (audioBuffer) {
                        newSound.setAudioBuffer(audioBuffer);
                    }
                    newSound._isReadyToPlay = true;
                    if (newSound.autoplay) {
                        newSound.play(0, sourceSound._optionsV2.startOffset, sourceSound._optionsV2.duration);
                    }
                }, undefined, 300);
            };
            newSound = new Sound(soundName, new ArrayBuffer(0), scene, null, options);
            setBufferAndRun();
        }
        if (parsedSound.position) {
            const soundPosition = Vector3.FromArray(parsedSound.position);
            newSound.setPosition(soundPosition);
        }
        if (parsedSound.isDirectional) {
            newSound.setDirectionalCone(parsedSound.coneInnerAngle || 360, parsedSound.coneOuterAngle || 360, parsedSound.coneOuterGain || 0);
            if (parsedSound.localDirectionToMesh) {
                const localDirectionToMesh = Vector3.FromArray(parsedSound.localDirectionToMesh);
                newSound.setLocalDirectionToMesh(localDirectionToMesh);
            }
        }
        if (parsedSound.connectedMeshId) {
            const connectedMesh = scene.getMeshById(parsedSound.connectedMeshId);
            if (connectedMesh) {
                newSound.attachToMesh(connectedMesh);
            }
        }
        if (parsedSound.metadata) {
            newSound.metadata = parsedSound.metadata;
        }
        return newSound;
    }
}
/**
 * @internal
 */
Sound._SceneComponentInitialization = (_) => {
    throw _WarnImport("AudioSceneComponent");
};
// Register Class Name
RegisterClass("BABYLON.Sound", Sound);

/**
 * Wraps one or more Sound objects and selects one with random weight for playback.
 */
class WeightedSound {
    /**
     * Creates a new WeightedSound from the list of sounds given.
     * @param loop When true a Sound will be selected and played when the current playing Sound completes.
     * @param sounds Array of Sounds that will be selected from.
     * @param weights Array of number values for selection weights; length must equal sounds, values will be normalized to 1
     */
    constructor(loop, sounds, weights) {
        /** When true a Sound will be selected and played when the current playing Sound completes. */
        this.loop = false;
        this._coneInnerAngle = 360;
        this._coneOuterAngle = 360;
        this._volume = 1;
        /** A Sound is currently playing. */
        this.isPlaying = false;
        /** A Sound is currently paused. */
        this.isPaused = false;
        this._sounds = [];
        this._weights = [];
        if (sounds.length !== weights.length) {
            throw new Error("Sounds length does not equal weights length");
        }
        this.loop = loop;
        this._weights = weights;
        // Normalize the weights
        let weightSum = 0;
        for (const weight of weights) {
            weightSum += weight;
        }
        const invWeightSum = weightSum > 0 ? 1 / weightSum : 0;
        for (let i = 0; i < this._weights.length; i++) {
            this._weights[i] *= invWeightSum;
        }
        this._sounds = sounds;
        for (const sound of this._sounds) {
            sound.onEndedObservable.add(() => {
                this._onended();
            });
        }
    }
    /**
     * The size of cone in degrees for a directional sound in which there will be no attenuation.
     */
    get directionalConeInnerAngle() {
        return this._coneInnerAngle;
    }
    /**
     * The size of cone in degrees for a directional sound in which there will be no attenuation.
     */
    set directionalConeInnerAngle(value) {
        if (value !== this._coneInnerAngle) {
            if (this._coneOuterAngle < value) {
                Logger.Error("directionalConeInnerAngle: outer angle of the cone must be superior or equal to the inner angle.");
                return;
            }
            this._coneInnerAngle = value;
            for (const sound of this._sounds) {
                sound.directionalConeInnerAngle = value;
            }
        }
    }
    /**
     * Size of cone in degrees for a directional sound outside of which there will be no sound.
     * Listener angles between innerAngle and outerAngle will falloff linearly.
     */
    get directionalConeOuterAngle() {
        return this._coneOuterAngle;
    }
    /**
     * Size of cone in degrees for a directional sound outside of which there will be no sound.
     * Listener angles between innerAngle and outerAngle will falloff linearly.
     */
    set directionalConeOuterAngle(value) {
        if (value !== this._coneOuterAngle) {
            if (value < this._coneInnerAngle) {
                Logger.Error("directionalConeOuterAngle: outer angle of the cone must be superior or equal to the inner angle.");
                return;
            }
            this._coneOuterAngle = value;
            for (const sound of this._sounds) {
                sound.directionalConeOuterAngle = value;
            }
        }
    }
    /**
     * Playback volume.
     */
    get volume() {
        return this._volume;
    }
    /**
     * Playback volume.
     */
    set volume(value) {
        if (value !== this._volume) {
            for (const sound of this._sounds) {
                sound.setVolume(value);
            }
        }
    }
    _onended() {
        if (this._currentIndex !== undefined) {
            this._sounds[this._currentIndex].autoplay = false;
        }
        if (this.loop && this.isPlaying) {
            this.play();
        }
        else {
            this.isPlaying = false;
        }
    }
    /**
     * Suspend playback
     */
    pause() {
        if (this.isPlaying) {
            this.isPaused = true;
            if (this._currentIndex !== undefined) {
                this._sounds[this._currentIndex].pause();
            }
        }
    }
    /**
     * Stop playback
     */
    stop() {
        this.isPlaying = false;
        if (this._currentIndex !== undefined) {
            this._sounds[this._currentIndex].stop();
        }
    }
    /**
     * Start playback.
     * @param startOffset Position the clip head at a specific time in seconds.
     */
    play(startOffset) {
        if (!this.isPaused) {
            this.stop();
            const randomValue = Math.random();
            let total = 0;
            for (let i = 0; i < this._weights.length; i++) {
                total += this._weights[i];
                if (randomValue <= total) {
                    this._currentIndex = i;
                    break;
                }
            }
        }
        const sound = this._sounds[this._currentIndex ?? 0];
        if (sound.isReady()) {
            sound.play(0, this.isPaused ? undefined : startOffset);
        }
        else {
            sound.autoplay = true;
        }
        this.isPlaying = true;
        this.isPaused = false;
    }
}

/**
 * It could be useful to isolate your music & sounds on several tracks to better manage volume on a grouped instance of sounds.
 * It will be also used in a future release to apply effects on a specific track.
 * @see https://doc.babylonjs.com/features/featuresDeepDive/audio/playingSoundsMusic#using-sound-tracks
 */
class SoundTrack {
    /**
     * Creates a new sound track.
     * @see https://doc.babylonjs.com/features/featuresDeepDive/audio/playingSoundsMusic#using-sound-tracks
     * @param scene Define the scene the sound track belongs to
     * @param options
     */
    constructor(scene, options = {}) {
        /**
         * The unique identifier of the sound track in the scene.
         */
        this.id = -1;
        this._isInitialized = false;
        scene = scene || EngineStore.LastCreatedScene;
        if (!scene) {
            return;
        }
        this._scene = scene;
        this.soundCollection = [];
        this._options = options;
        if (!this._options.mainTrack && this._scene.soundTracks) {
            this._scene.soundTracks.push(this);
            this.id = this._scene.soundTracks.length - 1;
        }
    }
    _initializeSoundTrackAudioGraph() {
        if (AbstractEngine.audioEngine?.canUseWebAudio && AbstractEngine.audioEngine.audioContext) {
            this._outputAudioNode = AbstractEngine.audioEngine.audioContext.createGain();
            this._outputAudioNode.connect(AbstractEngine.audioEngine.masterGain);
            if (this._options) {
                if (this._options.volume) {
                    this._outputAudioNode.gain.value = this._options.volume;
                }
            }
            this._isInitialized = true;
        }
    }
    /**
     * Release the sound track and its associated resources
     */
    dispose() {
        if (AbstractEngine.audioEngine && AbstractEngine.audioEngine.canUseWebAudio) {
            if (this._connectedAnalyser) {
                this._connectedAnalyser.stopDebugCanvas();
            }
            while (this.soundCollection.length) {
                this.soundCollection[0].dispose();
            }
            if (this._outputAudioNode) {
                this._outputAudioNode.disconnect();
            }
            this._outputAudioNode = null;
        }
    }
    /**
     * Adds a sound to this sound track
     * @param sound define the sound to add
     * @ignoreNaming
     */
    addSound(sound) {
        if (!this._isInitialized) {
            this._initializeSoundTrackAudioGraph();
        }
        if (AbstractEngine.audioEngine?.canUseWebAudio && this._outputAudioNode) {
            sound.connectToSoundTrackAudioNode(this._outputAudioNode);
        }
        if (sound.soundTrackId !== undefined) {
            if (sound.soundTrackId === -1) {
                this._scene.mainSoundTrack.removeSound(sound);
            }
            else if (this._scene.soundTracks) {
                this._scene.soundTracks[sound.soundTrackId].removeSound(sound);
            }
        }
        this.soundCollection.push(sound);
        sound.soundTrackId = this.id;
    }
    /**
     * Removes a sound to this sound track
     * @param sound define the sound to remove
     * @ignoreNaming
     */
    removeSound(sound) {
        const index = this.soundCollection.indexOf(sound);
        if (index !== -1) {
            this.soundCollection.splice(index, 1);
        }
    }
    /**
     * Set a global volume for the full sound track.
     * @param newVolume Define the new volume of the sound track
     */
    setVolume(newVolume) {
        if (AbstractEngine.audioEngine?.canUseWebAudio && this._outputAudioNode) {
            this._outputAudioNode.gain.value = newVolume;
        }
    }
    /**
     * Switch the panning model to HRTF:
     * Renders a stereo output of higher quality than equalpower — it uses a convolution with measured impulse responses from human subjects.
     * @see https://doc.babylonjs.com/features/featuresDeepDive/audio/playingSoundsMusic#creating-a-spatial-3d-sound
     */
    switchPanningModelToHRTF() {
        if (AbstractEngine.audioEngine?.canUseWebAudio) {
            for (let i = 0; i < this.soundCollection.length; i++) {
                this.soundCollection[i].switchPanningModelToHRTF();
            }
        }
    }
    /**
     * Switch the panning model to Equal Power:
     * Represents the equal-power panning algorithm, generally regarded as simple and efficient. equalpower is the default value.
     * @see https://doc.babylonjs.com/features/featuresDeepDive/audio/playingSoundsMusic#creating-a-spatial-3d-sound
     */
    switchPanningModelToEqualPower() {
        if (AbstractEngine.audioEngine?.canUseWebAudio) {
            for (let i = 0; i < this.soundCollection.length; i++) {
                this.soundCollection[i].switchPanningModelToEqualPower();
            }
        }
    }
    /**
     * Connect the sound track to an audio analyser allowing some amazing
     * synchronization between the sounds/music and your visualization (VuMeter for instance).
     * @see https://doc.babylonjs.com/features/featuresDeepDive/audio/playingSoundsMusic#using-the-analyser
     * @param analyser The analyser to connect to the engine
     */
    connectToAnalyser(analyser) {
        if (this._connectedAnalyser) {
            this._connectedAnalyser.stopDebugCanvas();
        }
        this._connectedAnalyser = analyser;
        if (AbstractEngine.audioEngine?.canUseWebAudio && this._outputAudioNode) {
            this._outputAudioNode.disconnect();
            this._connectedAnalyser.connectAudioNodes(this._outputAudioNode, AbstractEngine.audioEngine.masterGain);
        }
    }
}

Object.defineProperty(Scene.prototype, "mainSoundTrack", {
    get: function () {
        let compo = this._getComponent(SceneComponentConstants.NAME_AUDIO);
        if (!compo) {
            compo = new AudioSceneComponent(this);
            this._addComponent(compo);
        }
        if (!this._mainSoundTrack) {
            this._mainSoundTrack = new SoundTrack(this, { mainTrack: true });
        }
        return this._mainSoundTrack;
    },
    enumerable: true,
    configurable: true,
});
Scene.prototype.getSoundByName = function (name) {
    let index;
    for (index = 0; index < this.mainSoundTrack.soundCollection.length; index++) {
        if (this.mainSoundTrack.soundCollection[index].name === name) {
            return this.mainSoundTrack.soundCollection[index];
        }
    }
    if (this.soundTracks) {
        for (let sdIndex = 0; sdIndex < this.soundTracks.length; sdIndex++) {
            for (index = 0; index < this.soundTracks[sdIndex].soundCollection.length; index++) {
                if (this.soundTracks[sdIndex].soundCollection[index].name === name) {
                    return this.soundTracks[sdIndex].soundCollection[index];
                }
            }
        }
    }
    return null;
};
Object.defineProperty(Scene.prototype, "audioEnabled", {
    get: function () {
        let compo = this._getComponent(SceneComponentConstants.NAME_AUDIO);
        if (!compo) {
            compo = new AudioSceneComponent(this);
            this._addComponent(compo);
        }
        return compo.audioEnabled;
    },
    set: function (value) {
        let compo = this._getComponent(SceneComponentConstants.NAME_AUDIO);
        if (!compo) {
            compo = new AudioSceneComponent(this);
            this._addComponent(compo);
        }
        if (value) {
            compo.enableAudio();
        }
        else {
            compo.disableAudio();
        }
    },
    enumerable: true,
    configurable: true,
});
Object.defineProperty(Scene.prototype, "headphone", {
    get: function () {
        let compo = this._getComponent(SceneComponentConstants.NAME_AUDIO);
        if (!compo) {
            compo = new AudioSceneComponent(this);
            this._addComponent(compo);
        }
        return compo.headphone;
    },
    set: function (value) {
        let compo = this._getComponent(SceneComponentConstants.NAME_AUDIO);
        if (!compo) {
            compo = new AudioSceneComponent(this);
            this._addComponent(compo);
        }
        if (value) {
            compo.switchAudioModeForHeadphones();
        }
        else {
            compo.switchAudioModeForNormalSpeakers();
        }
    },
    enumerable: true,
    configurable: true,
});
Object.defineProperty(Scene.prototype, "audioListenerPositionProvider", {
    get: function () {
        let compo = this._getComponent(SceneComponentConstants.NAME_AUDIO);
        if (!compo) {
            compo = new AudioSceneComponent(this);
            this._addComponent(compo);
        }
        return compo.audioListenerPositionProvider;
    },
    set: function (value) {
        let compo = this._getComponent(SceneComponentConstants.NAME_AUDIO);
        if (!compo) {
            compo = new AudioSceneComponent(this);
            this._addComponent(compo);
        }
        if (value && typeof value !== "function") {
            throw new Error("The value passed to [Scene.audioListenerPositionProvider] must be a function that returns a Vector3");
        }
        else {
            compo.audioListenerPositionProvider = value;
        }
    },
    enumerable: true,
    configurable: true,
});
Object.defineProperty(Scene.prototype, "audioListenerRotationProvider", {
    get: function () {
        let compo = this._getComponent(SceneComponentConstants.NAME_AUDIO);
        if (!compo) {
            compo = new AudioSceneComponent(this);
            this._addComponent(compo);
        }
        return compo.audioListenerRotationProvider;
    },
    set: function (value) {
        let compo = this._getComponent(SceneComponentConstants.NAME_AUDIO);
        if (!compo) {
            compo = new AudioSceneComponent(this);
            this._addComponent(compo);
        }
        if (value && typeof value !== "function") {
            throw new Error("The value passed to [Scene.audioListenerRotationProvider] must be a function that returns a Vector3");
        }
        else {
            compo.audioListenerRotationProvider = value;
        }
    },
    enumerable: true,
    configurable: true,
});
Object.defineProperty(Scene.prototype, "audioPositioningRefreshRate", {
    get: function () {
        let compo = this._getComponent(SceneComponentConstants.NAME_AUDIO);
        if (!compo) {
            compo = new AudioSceneComponent(this);
            this._addComponent(compo);
        }
        return compo.audioPositioningRefreshRate;
    },
    set: function (value) {
        let compo = this._getComponent(SceneComponentConstants.NAME_AUDIO);
        if (!compo) {
            compo = new AudioSceneComponent(this);
            this._addComponent(compo);
        }
        compo.audioPositioningRefreshRate = value;
    },
    enumerable: true,
    configurable: true,
});
/**
 * Defines the sound scene component responsible to manage any sounds
 * in a given scene.
 * @deprecated please use AudioEngineV2 instead
 */
class AudioSceneComponent {
    /**
     * Gets whether audio is enabled or not.
     * Please use related enable/disable method to switch state.
     */
    get audioEnabled() {
        return this._audioEnabled;
    }
    /**
     * Gets whether audio is outputting to headphone or not.
     * Please use the according Switch methods to change output.
     */
    get headphone() {
        return this._headphone;
    }
    /**
     * Creates a new instance of the component for the given scene
     * @param scene Defines the scene to register the component in
     */
    constructor(scene) {
        /**
         * The component name helpful to identify the component in the list of scene components.
         */
        this.name = SceneComponentConstants.NAME_AUDIO;
        this._audioEnabled = true;
        this._headphone = false;
        /**
         * Gets or sets a refresh rate when using 3D audio positioning
         */
        this.audioPositioningRefreshRate = 500;
        /**
         * Gets or Sets a custom listener position for all sounds in the scene
         * By default, this is the position of the first active camera
         */
        this.audioListenerPositionProvider = null;
        /**
         * Gets or Sets a custom listener rotation for all sounds in the scene
         * By default, this is the rotation of the first active camera
         */
        this.audioListenerRotationProvider = null;
        this._cachedCameraDirection = new Vector3();
        this._cachedCameraPosition = new Vector3();
        this._lastCheck = 0;
        this._invertMatrixTemp = new Matrix();
        this._cameraDirectionTemp = new Vector3();
        scene = scene || EngineStore.LastCreatedScene;
        if (!scene) {
            return;
        }
        this.scene = scene;
        scene.soundTracks = [];
        scene.sounds = [];
    }
    /**
     * Registers the component in a given scene
     */
    register() {
        this.scene._afterRenderStage.registerStep(SceneComponentConstants.STEP_AFTERRENDER_AUDIO, this, this._afterRender);
    }
    /**
     * Rebuilds the elements related to this component in case of
     * context lost for instance.
     */
    rebuild() {
        // Nothing to do here. (Not rendering related)
    }
    /**
     * Serializes the component data to the specified json object
     * @param serializationObject The object to serialize to
     */
    serialize(serializationObject) {
        serializationObject.sounds = [];
        if (this.scene.soundTracks) {
            for (let index = 0; index < this.scene.soundTracks.length; index++) {
                const soundtrack = this.scene.soundTracks[index];
                for (let soundId = 0; soundId < soundtrack.soundCollection.length; soundId++) {
                    serializationObject.sounds.push(soundtrack.soundCollection[soundId].serialize());
                }
            }
        }
    }
    /**
     * Adds all the elements from the container to the scene
     * @param container the container holding the elements
     */
    addFromContainer(container) {
        if (!container.sounds) {
            return;
        }
        for (const sound of container.sounds) {
            sound.play();
            sound.autoplay = true;
            this.scene.mainSoundTrack.addSound(sound);
        }
    }
    /**
     * Removes all the elements in the container from the scene
     * @param container contains the elements to remove
     * @param dispose if the removed element should be disposed (default: false)
     */
    removeFromContainer(container, dispose = false) {
        if (!container.sounds) {
            return;
        }
        for (const sound of container.sounds) {
            sound.stop();
            sound.autoplay = false;
            this.scene.mainSoundTrack.removeSound(sound);
            if (dispose) {
                sound.dispose();
            }
        }
    }
    /**
     * Disposes the component and the associated resources.
     */
    dispose() {
        const scene = this.scene;
        if (scene._mainSoundTrack) {
            scene.mainSoundTrack.dispose();
        }
        if (scene.soundTracks) {
            for (let scIndex = 0; scIndex < scene.soundTracks.length; scIndex++) {
                scene.soundTracks[scIndex].dispose();
            }
        }
    }
    /**
     * Disables audio in the associated scene.
     */
    disableAudio() {
        const scene = this.scene;
        this._audioEnabled = false;
        if (AbstractEngine.audioEngine && AbstractEngine.audioEngine.audioContext) {
            // eslint-disable-next-line @typescript-eslint/no-floating-promises
            AbstractEngine.audioEngine.audioContext.suspend();
        }
        let i;
        for (i = 0; i < scene.mainSoundTrack.soundCollection.length; i++) {
            scene.mainSoundTrack.soundCollection[i].pause();
        }
        if (scene.soundTracks) {
            for (i = 0; i < scene.soundTracks.length; i++) {
                for (let j = 0; j < scene.soundTracks[i].soundCollection.length; j++) {
                    scene.soundTracks[i].soundCollection[j].pause();
                }
            }
        }
    }
    /**
     * Enables audio in the associated scene.
     */
    enableAudio() {
        const scene = this.scene;
        this._audioEnabled = true;
        if (AbstractEngine.audioEngine && AbstractEngine.audioEngine.audioContext) {
            // eslint-disable-next-line @typescript-eslint/no-floating-promises
            AbstractEngine.audioEngine.audioContext.resume();
        }
        let i;
        for (i = 0; i < scene.mainSoundTrack.soundCollection.length; i++) {
            if (scene.mainSoundTrack.soundCollection[i].isPaused) {
                scene.mainSoundTrack.soundCollection[i].play();
            }
        }
        if (scene.soundTracks) {
            for (i = 0; i < scene.soundTracks.length; i++) {
                for (let j = 0; j < scene.soundTracks[i].soundCollection.length; j++) {
                    if (scene.soundTracks[i].soundCollection[j].isPaused) {
                        scene.soundTracks[i].soundCollection[j].play();
                    }
                }
            }
        }
    }
    /**
     * Switch audio to headphone output.
     */
    switchAudioModeForHeadphones() {
        const scene = this.scene;
        this._headphone = true;
        scene.mainSoundTrack.switchPanningModelToHRTF();
        if (scene.soundTracks) {
            for (let i = 0; i < scene.soundTracks.length; i++) {
                scene.soundTracks[i].switchPanningModelToHRTF();
            }
        }
    }
    /**
     * Switch audio to normal speakers.
     */
    switchAudioModeForNormalSpeakers() {
        const scene = this.scene;
        this._headphone = false;
        scene.mainSoundTrack.switchPanningModelToEqualPower();
        if (scene.soundTracks) {
            for (let i = 0; i < scene.soundTracks.length; i++) {
                scene.soundTracks[i].switchPanningModelToEqualPower();
            }
        }
    }
    _afterRender() {
        const now = PrecisionDate.Now;
        if (this._lastCheck && now - this._lastCheck < this.audioPositioningRefreshRate) {
            return;
        }
        this._lastCheck = now;
        const scene = this.scene;
        if (!this._audioEnabled || !scene._mainSoundTrack || !scene.soundTracks || (scene._mainSoundTrack.soundCollection.length === 0 && scene.soundTracks.length === 1)) {
            return;
        }
        const audioEngine = AbstractEngine.audioEngine;
        if (!audioEngine) {
            return;
        }
        if (audioEngine.audioContext) {
            let listeningCamera = scene.activeCamera;
            if (scene.activeCameras && scene.activeCameras.length > 0) {
                listeningCamera = scene.activeCameras[0];
            }
            // A custom listener position provider was set
            // Use the users provided position instead of camera's
            if (this.audioListenerPositionProvider) {
                const position = this.audioListenerPositionProvider();
                // Set the listener position
                audioEngine.audioContext.listener.setPosition(position.x || 0, position.y || 0, position.z || 0);
                // Check if there is a listening camera
            }
            else if (listeningCamera) {
                // Set the listener position to the listening camera global position
                if (!this._cachedCameraPosition.equals(listeningCamera.globalPosition)) {
                    this._cachedCameraPosition.copyFrom(listeningCamera.globalPosition);
                    audioEngine.audioContext.listener.setPosition(listeningCamera.globalPosition.x, listeningCamera.globalPosition.y, listeningCamera.globalPosition.z);
                }
            }
            // Otherwise set the listener position to 0, 0 ,0
            else {
                // Set the listener position
                audioEngine.audioContext.listener.setPosition(0, 0, 0);
            }
            // A custom listener rotation provider was set
            // Use the users provided rotation instead of camera's
            if (this.audioListenerRotationProvider) {
                const rotation = this.audioListenerRotationProvider();
                audioEngine.audioContext.listener.setOrientation(rotation.x || 0, rotation.y || 0, rotation.z || 0, 0, 1, 0);
                // Check if there is a listening camera
            }
            else if (listeningCamera) {
                // for VR cameras
                if (listeningCamera.rigCameras && listeningCamera.rigCameras.length > 0) {
                    listeningCamera = listeningCamera.rigCameras[0];
                }
                listeningCamera.getViewMatrix().invertToRef(this._invertMatrixTemp);
                Vector3.TransformNormalToRef(AudioSceneComponent._CameraDirection, this._invertMatrixTemp, this._cameraDirectionTemp);
                this._cameraDirectionTemp.normalize();
                // To avoid some errors on GearVR
                if (!isNaN(this._cameraDirectionTemp.x) && !isNaN(this._cameraDirectionTemp.y) && !isNaN(this._cameraDirectionTemp.z)) {
                    if (!this._cachedCameraDirection.equals(this._cameraDirectionTemp)) {
                        this._cachedCameraDirection.copyFrom(this._cameraDirectionTemp);
                        audioEngine.audioContext.listener.setOrientation(this._cameraDirectionTemp.x, this._cameraDirectionTemp.y, this._cameraDirectionTemp.z, 0, 1, 0);
                    }
                }
            }
            // Otherwise set the listener rotation to 0, 0 ,0
            else {
                // Set the listener position
                audioEngine.audioContext.listener.setOrientation(0, 0, 0, 0, 1, 0);
            }
            let i;
            for (i = 0; i < scene.mainSoundTrack.soundCollection.length; i++) {
                const sound = scene.mainSoundTrack.soundCollection[i];
                if (sound.useCustomAttenuation) {
                    sound.updateDistanceFromListener();
                }
            }
            if (scene.soundTracks) {
                for (i = 0; i < scene.soundTracks.length; i++) {
                    for (let j = 0; j < scene.soundTracks[i].soundCollection.length; j++) {
                        const sound = scene.soundTracks[i].soundCollection[j];
                        if (sound.useCustomAttenuation) {
                            sound.updateDistanceFromListener();
                        }
                    }
                }
            }
        }
    }
}
AudioSceneComponent._CameraDirection = new Vector3(0, 0, -1);
Sound._SceneComponentInitialization = (scene) => {
    let compo = scene._getComponent(SceneComponentConstants.NAME_AUDIO);
    if (!compo) {
        compo = new AudioSceneComponent(scene);
        scene._addComponent(compo);
    }
};

const NAME = "MSFT_audio_emitter";
/**
 * [Specification](https://github.com/najadojo/glTF/blob/MSFT_audio_emitter/extensions/2.0/Vendor/MSFT_audio_emitter/README.md)
 * !!! Experimental Extension Subject to Changes !!!
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
class MSFT_audio_emitter {
    /**
     * @internal
     */
    constructor(loader) {
        /**
         * The name of this extension.
         */
        this.name = NAME;
        this._loader = loader;
        this.enabled = this._loader.isExtensionUsed(NAME);
    }
    /** @internal */
    dispose() {
        this._loader = null;
        this._clips = null;
        this._emitters = null;
    }
    /** @internal */
    onLoading() {
        const extensions = this._loader.gltf.extensions;
        if (extensions && extensions[this.name]) {
            const extension = extensions[this.name];
            this._clips = extension.clips;
            this._emitters = extension.emitters;
            ArrayItem.Assign(this._clips);
            ArrayItem.Assign(this._emitters);
        }
    }
    /**
     * @internal
     */
    // eslint-disable-next-line no-restricted-syntax
    loadSceneAsync(context, scene) {
        return GLTFLoader.LoadExtensionAsync(context, scene, this.name, async (extensionContext, extension) => {
            const promises = new Array();
            promises.push(this._loader.loadSceneAsync(context, scene));
            for (const emitterIndex of extension.emitters) {
                const emitter = ArrayItem.Get(`${extensionContext}/emitters`, this._emitters, emitterIndex);
                if (emitter.refDistance != undefined ||
                    emitter.maxDistance != undefined ||
                    emitter.rolloffFactor != undefined ||
                    emitter.distanceModel != undefined ||
                    emitter.innerAngle != undefined ||
                    emitter.outerAngle != undefined) {
                    throw new Error(`${extensionContext}: Direction or Distance properties are not allowed on emitters attached to a scene`);
                }
                promises.push(this._loadEmitterAsync(`${extensionContext}/emitters/${emitter.index}`, emitter));
            }
            await Promise.all(promises);
        });
    }
    /**
     * @internal
     */
    // eslint-disable-next-line no-restricted-syntax
    loadNodeAsync(context, node, assign) {
        return GLTFLoader.LoadExtensionAsync(context, node, this.name, async (extensionContext, extension) => {
            const promises = new Array();
            const babylonMesh = await this._loader.loadNodeAsync(extensionContext, node, (babylonMesh) => {
                for (const emitterIndex of extension.emitters) {
                    const emitter = ArrayItem.Get(`${extensionContext}/emitters`, this._emitters, emitterIndex);
                    promises.push(
                    // eslint-disable-next-line github/no-then
                    this._loadEmitterAsync(`${extensionContext}/emitters/${emitter.index}`, emitter).then(() => {
                        for (const sound of emitter._babylonSounds) {
                            sound.attachToMesh(babylonMesh);
                            if (emitter.innerAngle != undefined || emitter.outerAngle != undefined) {
                                sound.setLocalDirectionToMesh(Vector3.Forward());
                                sound.setDirectionalCone(2 * Tools.ToDegrees(emitter.innerAngle == undefined ? Math.PI : emitter.innerAngle), 2 * Tools.ToDegrees(emitter.outerAngle == undefined ? Math.PI : emitter.outerAngle), 0);
                            }
                        }
                    }));
                }
                assign(babylonMesh);
            });
            await Promise.all(promises);
            return babylonMesh;
        });
    }
    /**
     * @internal
     */
    // eslint-disable-next-line no-restricted-syntax
    loadAnimationAsync(context, animation) {
        return GLTFLoader.LoadExtensionAsync(context, animation, this.name, async (extensionContext, extension) => {
            const babylonAnimationGroup = await this._loader.loadAnimationAsync(context, animation);
            const promises = new Array();
            ArrayItem.Assign(extension.events);
            for (const event of extension.events) {
                promises.push(this._loadAnimationEventAsync(`${extensionContext}/events/${event.index}`, context, animation, event, babylonAnimationGroup));
            }
            await Promise.all(promises);
            return babylonAnimationGroup;
        });
    }
    // eslint-disable-next-line @typescript-eslint/promise-function-async, no-restricted-syntax
    _loadClipAsync(context, clip) {
        if (clip._objectURL) {
            return clip._objectURL;
        }
        let promise;
        if (clip.uri) {
            promise = this._loader.loadUriAsync(context, clip, clip.uri);
        }
        else {
            const bufferView = ArrayItem.Get(`${context}/bufferView`, this._loader.gltf.bufferViews, clip.bufferView);
            promise = this._loader.loadBufferViewAsync(`/bufferViews/${bufferView.index}`, bufferView);
        }
        // eslint-disable-next-line github/no-then
        clip._objectURL = promise.then((data) => {
            return URL.createObjectURL(new Blob([data], { type: clip.mimeType }));
        });
        return clip._objectURL;
    }
    // eslint-disable-next-line @typescript-eslint/promise-function-async, no-restricted-syntax
    _loadEmitterAsync(context, emitter) {
        emitter._babylonSounds = emitter._babylonSounds || [];
        if (!emitter._babylonData) {
            const clipPromises = new Array();
            const name = emitter.name || `emitter${emitter.index}`;
            const options = {
                loop: false,
                autoplay: false,
                volume: emitter.volume == undefined ? 1 : emitter.volume,
            };
            for (let i = 0; i < emitter.clips.length; i++) {
                const clipContext = `/extensions/${this.name}/clips`;
                const clip = ArrayItem.Get(clipContext, this._clips, emitter.clips[i].clip);
                clipPromises.push(
                // eslint-disable-next-line github/no-then
                this._loadClipAsync(`${clipContext}/${emitter.clips[i].clip}`, clip).then((objectURL) => {
                    const sound = (emitter._babylonSounds[i] = new Sound(name, objectURL, this._loader.babylonScene, null, options));
                    sound.refDistance = emitter.refDistance || 1;
                    sound.maxDistance = emitter.maxDistance || 256;
                    sound.rolloffFactor = emitter.rolloffFactor || 1;
                    sound.distanceModel = emitter.distanceModel || "exponential";
                }));
            }
            // eslint-disable-next-line github/no-then
            const promise = Promise.all(clipPromises).then(() => {
                const weights = emitter.clips.map((clip) => {
                    return clip.weight || 1;
                });
                const weightedSound = new WeightedSound(emitter.loop || false, emitter._babylonSounds, weights);
                if (emitter.innerAngle) {
                    weightedSound.directionalConeInnerAngle = 2 * Tools.ToDegrees(emitter.innerAngle);
                }
                if (emitter.outerAngle) {
                    weightedSound.directionalConeOuterAngle = 2 * Tools.ToDegrees(emitter.outerAngle);
                }
                if (emitter.volume) {
                    weightedSound.volume = emitter.volume;
                }
                emitter._babylonData.sound = weightedSound;
            });
            emitter._babylonData = {
                loaded: promise,
            };
        }
        return emitter._babylonData.loaded;
    }
    _getEventAction(context, sound, action, time, startOffset) {
        switch (action) {
            case "play" /* IMSFTAudioEmitter_AnimationEventAction.play */: {
                return (currentFrame) => {
                    const frameOffset = (startOffset || 0) + (currentFrame - time);
                    sound.play(frameOffset);
                };
            }
            case "stop" /* IMSFTAudioEmitter_AnimationEventAction.stop */: {
                return () => {
                    sound.stop();
                };
            }
            case "pause" /* IMSFTAudioEmitter_AnimationEventAction.pause */: {
                return () => {
                    sound.pause();
                };
            }
            default: {
                throw new Error(`${context}: Unsupported action ${action}`);
            }
        }
    }
    // eslint-disable-next-line @typescript-eslint/promise-function-async, no-restricted-syntax
    _loadAnimationEventAsync(context, animationContext, animation, event, babylonAnimationGroup) {
        if (babylonAnimationGroup.targetedAnimations.length == 0) {
            return Promise.resolve();
        }
        const babylonAnimation = babylonAnimationGroup.targetedAnimations[0];
        const emitterIndex = event.emitter;
        const emitter = ArrayItem.Get(`/extensions/${this.name}/emitters`, this._emitters, emitterIndex);
        // eslint-disable-next-line github/no-then
        return this._loadEmitterAsync(context, emitter).then(() => {
            const sound = emitter._babylonData.sound;
            if (sound) {
                const babylonAnimationEvent = new AnimationEvent(event.time, this._getEventAction(context, sound, event.action, event.time, event.startOffset));
                babylonAnimation.animation.addEvent(babylonAnimationEvent);
                // Make sure all started audio stops when this animation is terminated.
                babylonAnimationGroup.onAnimationGroupEndObservable.add(() => {
                    sound.stop();
                });
                babylonAnimationGroup.onAnimationGroupPauseObservable.add(() => {
                    sound.pause();
                });
            }
        });
    }
}
unregisterGLTFExtension(NAME);
registerGLTFExtension(NAME, true, (loader) => new MSFT_audio_emitter(loader));

export { MSFT_audio_emitter };
//# sourceMappingURL=MSFT_audio_emitter-CFOOLenC.js.map
