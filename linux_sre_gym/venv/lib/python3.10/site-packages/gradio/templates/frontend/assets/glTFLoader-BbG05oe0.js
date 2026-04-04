const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["./animationGroup-CGqF01WM.js","./index-CxR0Yf_7.js","./index-BvBk1Iap.js","./index-CrbFh6qt.css","./bone-C6xIrJX4.js","./glTFLoaderAnimation-DNslULmB.js","./objectModelMapping-DnUo8NMz.js","./openpbrMaterial-L3-DYNd8.js","./pbrMaterialLoadingAdapter-vcu2PXuo.js"])))=>i.map(i=>d[i]);
import { af as __vitePreload } from './index-BvBk1Iap.js';
import { d as Tools, bG as KeyboardEventTypes, V as Vector3, _ as __decorate, s as serialize, bH as CameraInputTypes, O as Observable, aQ as PointerEventTypes, bI as EventConstants, M as Matrix, bJ as CameraInputsManager, bK as TargetCamera, A as AbstractEngine, ah as Vector2, f as serializeAsVector3, R as RegisterClass, aq as EngineStore, e as SerializationHelper, G as GetClass, J as VertexBuffer, T as Texture, n as SmartArray, L as Logger, bL as Decode, bM as GLTFFileLoaderMetadata, bN as GLTFMagicBase64Encoded, bO as DecodeBase64UrlToBinary, bP as RuntimeError, bQ as ErrorCodes, ax as RegisterSceneLoaderPlugin, aH as registerGLTFExtension, aG as unregisterGLTFExtension, o as Material, bR as registeredGLTFExtensions, l as Mesh, bg as AbstractMesh, bh as TransformNode, bS as deepMerge, bp as Geometry, ao as Quaternion, b1 as Camera, bT as Lazy, au as Buffer, C as Color3, aM as Deferred, bU as GetMimeType, bV as IsBase64DataUrl, bW as LoadFileError, bX as GetTypedArrayConstructor, c as TmpVectors, av as BoundingInfo } from './index-CxR0Yf_7.js';
import { B as Bone } from './bone-C6xIrJX4.js';
import { S as Skeleton } from './skeleton-DzANFt7R.js';
import { A as AssetContainer } from './assetContainer-7GPvH2S_.js';
import { b as GetMappingForKey } from './objectModelMapping-DnUo8NMz.js';
import './rawTexture-Bto4WZO2.js';

/**
 * Manage the keyboard inputs to control the movement of a free camera.
 * @see https://doc.babylonjs.com/features/featuresDeepDive/cameras/customizingCameraInputs
 */
class FreeCameraKeyboardMoveInput {
    constructor() {
        /**
         * Gets or Set the list of keyboard keys used to control the forward move of the camera.
         */
        this.keysUp = [38];
        /**
         * Gets or Set the list of keyboard keys used to control the upward move of the camera.
         */
        this.keysUpward = [33];
        /**
         * Gets or Set the list of keyboard keys used to control the backward move of the camera.
         */
        this.keysDown = [40];
        /**
         * Gets or Set the list of keyboard keys used to control the downward move of the camera.
         */
        this.keysDownward = [34];
        /**
         * Gets or Set the list of keyboard keys used to control the left strafe move of the camera.
         */
        this.keysLeft = [37];
        /**
         * Gets or Set the list of keyboard keys used to control the right strafe move of the camera.
         */
        this.keysRight = [39];
        /**
         * Defines the pointer angular sensibility  along the X and Y axis or how fast is the camera rotating.
         */
        this.rotationSpeed = 0.5;
        /**
         * Gets or Set the list of keyboard keys used to control the left rotation move of the camera.
         */
        this.keysRotateLeft = [];
        /**
         * Gets or Set the list of keyboard keys used to control the right rotation move of the camera.
         */
        this.keysRotateRight = [];
        /**
         * Gets or Set the list of keyboard keys used to control the up rotation move of the camera.
         */
        this.keysRotateUp = [];
        /**
         * Gets or Set the list of keyboard keys used to control the down rotation move of the camera.
         */
        this.keysRotateDown = [];
        this._keys = new Array();
    }
    /**
     * Attach the input controls to a specific dom element to get the input from.
     * @param noPreventDefault Defines whether event caught by the controls should call preventdefault() (https://developer.mozilla.org/en-US/docs/Web/API/Event/preventDefault)
     */
    attachControl(noPreventDefault) {
        noPreventDefault = Tools.BackCompatCameraNoPreventDefault(arguments);
        if (this._onCanvasBlurObserver) {
            return;
        }
        this._scene = this.camera.getScene();
        this._engine = this._scene.getEngine();
        this._onCanvasBlurObserver = this._engine.onCanvasBlurObservable.add(() => {
            this._keys.length = 0;
        });
        this._onKeyboardObserver = this._scene.onKeyboardObservable.add((info) => {
            const evt = info.event;
            if (!evt.metaKey) {
                if (info.type === KeyboardEventTypes.KEYDOWN) {
                    if (this.keysUp.indexOf(evt.keyCode) !== -1 ||
                        this.keysDown.indexOf(evt.keyCode) !== -1 ||
                        this.keysLeft.indexOf(evt.keyCode) !== -1 ||
                        this.keysRight.indexOf(evt.keyCode) !== -1 ||
                        this.keysUpward.indexOf(evt.keyCode) !== -1 ||
                        this.keysDownward.indexOf(evt.keyCode) !== -1 ||
                        this.keysRotateLeft.indexOf(evt.keyCode) !== -1 ||
                        this.keysRotateRight.indexOf(evt.keyCode) !== -1 ||
                        this.keysRotateUp.indexOf(evt.keyCode) !== -1 ||
                        this.keysRotateDown.indexOf(evt.keyCode) !== -1) {
                        const index = this._keys.indexOf(evt.keyCode);
                        if (index === -1) {
                            this._keys.push(evt.keyCode);
                        }
                        if (!noPreventDefault) {
                            evt.preventDefault();
                        }
                    }
                }
                else {
                    if (this.keysUp.indexOf(evt.keyCode) !== -1 ||
                        this.keysDown.indexOf(evt.keyCode) !== -1 ||
                        this.keysLeft.indexOf(evt.keyCode) !== -1 ||
                        this.keysRight.indexOf(evt.keyCode) !== -1 ||
                        this.keysUpward.indexOf(evt.keyCode) !== -1 ||
                        this.keysDownward.indexOf(evt.keyCode) !== -1 ||
                        this.keysRotateLeft.indexOf(evt.keyCode) !== -1 ||
                        this.keysRotateRight.indexOf(evt.keyCode) !== -1 ||
                        this.keysRotateUp.indexOf(evt.keyCode) !== -1 ||
                        this.keysRotateDown.indexOf(evt.keyCode) !== -1) {
                        const index = this._keys.indexOf(evt.keyCode);
                        if (index >= 0) {
                            this._keys.splice(index, 1);
                        }
                        if (!noPreventDefault) {
                            evt.preventDefault();
                        }
                    }
                }
            }
        });
    }
    /**
     * Detach the current controls from the specified dom element.
     */
    detachControl() {
        if (this._scene) {
            if (this._onKeyboardObserver) {
                this._scene.onKeyboardObservable.remove(this._onKeyboardObserver);
            }
            if (this._onCanvasBlurObserver) {
                this._engine.onCanvasBlurObservable.remove(this._onCanvasBlurObserver);
            }
            this._onKeyboardObserver = null;
            this._onCanvasBlurObserver = null;
        }
        this._keys.length = 0;
    }
    /**
     * Update the current camera state depending on the inputs that have been used this frame.
     * This is a dynamically created lambda to avoid the performance penalty of looping for inputs in the render loop.
     */
    checkInputs() {
        if (this._onKeyboardObserver) {
            const camera = this.camera;
            // Keyboard
            for (let index = 0; index < this._keys.length; index++) {
                const keyCode = this._keys[index];
                const speed = camera._computeLocalCameraSpeed();
                if (this.keysLeft.indexOf(keyCode) !== -1) {
                    camera._localDirection.copyFromFloats(-speed, 0, 0);
                }
                else if (this.keysUp.indexOf(keyCode) !== -1) {
                    camera._localDirection.copyFromFloats(0, 0, speed);
                }
                else if (this.keysRight.indexOf(keyCode) !== -1) {
                    camera._localDirection.copyFromFloats(speed, 0, 0);
                }
                else if (this.keysDown.indexOf(keyCode) !== -1) {
                    camera._localDirection.copyFromFloats(0, 0, -speed);
                }
                else if (this.keysUpward.indexOf(keyCode) !== -1) {
                    camera._localDirection.copyFromFloats(0, speed, 0);
                }
                else if (this.keysDownward.indexOf(keyCode) !== -1) {
                    camera._localDirection.copyFromFloats(0, -speed, 0);
                }
                else if (this.keysRotateLeft.indexOf(keyCode) !== -1) {
                    camera._localDirection.copyFromFloats(0, 0, 0);
                    camera.cameraRotation.y -= this._getLocalRotation();
                }
                else if (this.keysRotateRight.indexOf(keyCode) !== -1) {
                    camera._localDirection.copyFromFloats(0, 0, 0);
                    camera.cameraRotation.y += this._getLocalRotation();
                }
                else if (this.keysRotateUp.indexOf(keyCode) !== -1) {
                    camera._localDirection.copyFromFloats(0, 0, 0);
                    camera.cameraRotation.x -= this._getLocalRotation();
                }
                else if (this.keysRotateDown.indexOf(keyCode) !== -1) {
                    camera._localDirection.copyFromFloats(0, 0, 0);
                    camera.cameraRotation.x += this._getLocalRotation();
                }
                if (camera.getScene().useRightHandedSystem) {
                    camera._localDirection.z *= -1;
                }
                camera.getViewMatrix().invertToRef(camera._cameraTransformMatrix);
                Vector3.TransformNormalToRef(camera._localDirection, camera._cameraTransformMatrix, camera._transformedDirection);
                camera.cameraDirection.addInPlace(camera._transformedDirection);
            }
        }
    }
    /**
     * Gets the class name of the current input.
     * @returns the class name
     */
    getClassName() {
        return "FreeCameraKeyboardMoveInput";
    }
    /** @internal */
    _onLostFocus() {
        this._keys.length = 0;
    }
    /**
     * Get the friendly name associated with the input class.
     * @returns the input friendly name
     */
    getSimpleName() {
        return "keyboard";
    }
    _getLocalRotation() {
        const handednessMultiplier = this.camera._calculateHandednessMultiplier();
        const rotation = ((this.rotationSpeed * this._engine.getDeltaTime()) / 1000) * handednessMultiplier;
        return rotation;
    }
}
__decorate([
    serialize()
], FreeCameraKeyboardMoveInput.prototype, "keysUp", void 0);
__decorate([
    serialize()
], FreeCameraKeyboardMoveInput.prototype, "keysUpward", void 0);
__decorate([
    serialize()
], FreeCameraKeyboardMoveInput.prototype, "keysDown", void 0);
__decorate([
    serialize()
], FreeCameraKeyboardMoveInput.prototype, "keysDownward", void 0);
__decorate([
    serialize()
], FreeCameraKeyboardMoveInput.prototype, "keysLeft", void 0);
__decorate([
    serialize()
], FreeCameraKeyboardMoveInput.prototype, "keysRight", void 0);
__decorate([
    serialize()
], FreeCameraKeyboardMoveInput.prototype, "rotationSpeed", void 0);
__decorate([
    serialize()
], FreeCameraKeyboardMoveInput.prototype, "keysRotateLeft", void 0);
__decorate([
    serialize()
], FreeCameraKeyboardMoveInput.prototype, "keysRotateRight", void 0);
__decorate([
    serialize()
], FreeCameraKeyboardMoveInput.prototype, "keysRotateUp", void 0);
__decorate([
    serialize()
], FreeCameraKeyboardMoveInput.prototype, "keysRotateDown", void 0);
CameraInputTypes["FreeCameraKeyboardMoveInput"] = FreeCameraKeyboardMoveInput;

/**
 * Manage the mouse inputs to control the movement of a free camera.
 * @see https://doc.babylonjs.com/features/featuresDeepDive/cameras/customizingCameraInputs
 */
class FreeCameraMouseInput {
    /**
     * Manage the mouse inputs to control the movement of a free camera.
     * @see https://doc.babylonjs.com/features/featuresDeepDive/cameras/customizingCameraInputs
     * @param touchEnabled Defines if touch is enabled or not
     */
    constructor(
    /**
     * [true] Define if touch is enabled in the mouse input
     */
    touchEnabled = true) {
        this.touchEnabled = touchEnabled;
        /**
         * Defines the buttons associated with the input to handle camera move.
         */
        this.buttons = [0, 1, 2];
        /**
         * Defines the pointer angular sensibility  along the X and Y axis or how fast is the camera rotating.
         */
        this.angularSensibility = 2000.0;
        this._previousPosition = null;
        /**
         * Observable for when a pointer move event occurs containing the move offset
         */
        this.onPointerMovedObservable = new Observable();
        /**
         * @internal
         * If the camera should be rotated automatically based on pointer movement
         */
        this._allowCameraRotation = true;
        this._currentActiveButton = -1;
        this._activePointerId = -1;
    }
    /**
     * Attach the input controls to a specific dom element to get the input from.
     * @param noPreventDefault Defines whether event caught by the controls should call preventdefault() (https://developer.mozilla.org/en-US/docs/Web/API/Event/preventDefault)
     */
    attachControl(noPreventDefault) {
        noPreventDefault = Tools.BackCompatCameraNoPreventDefault(arguments);
        const engine = this.camera.getEngine();
        const element = engine.getInputElement();
        if (!this._pointerInput) {
            this._pointerInput = (p) => {
                const evt = p.event;
                const isTouch = evt.pointerType === "touch";
                if (!this.touchEnabled && isTouch) {
                    return;
                }
                if (p.type !== PointerEventTypes.POINTERMOVE && this.buttons.indexOf(evt.button) === -1) {
                    return;
                }
                const srcElement = evt.target;
                if (p.type === PointerEventTypes.POINTERDOWN) {
                    // If the input is touch with more than one touch OR if the input is mouse and there is already an active button, return
                    if ((isTouch && this._activePointerId !== -1) || (!isTouch && this._currentActiveButton !== -1)) {
                        return;
                    }
                    this._activePointerId = evt.pointerId;
                    try {
                        srcElement?.setPointerCapture(evt.pointerId);
                    }
                    catch (e) {
                        //Nothing to do with the error. Execution will continue.
                    }
                    if (this._currentActiveButton === -1) {
                        this._currentActiveButton = evt.button;
                    }
                    this._previousPosition = {
                        x: evt.clientX,
                        y: evt.clientY,
                    };
                    if (!noPreventDefault) {
                        evt.preventDefault();
                        if (element) {
                            element.focus();
                        }
                    }
                    // This is required to move while pointer button is down
                    if (engine.isPointerLock && this._onMouseMove) {
                        this._onMouseMove(p.event);
                    }
                }
                else if (p.type === PointerEventTypes.POINTERUP) {
                    // If input is touch with a different touch id OR if input is mouse with a different button, return
                    if ((isTouch && this._activePointerId !== evt.pointerId) || (!isTouch && this._currentActiveButton !== evt.button)) {
                        return;
                    }
                    try {
                        srcElement?.releasePointerCapture(evt.pointerId);
                    }
                    catch (e) {
                        //Nothing to do with the error.
                    }
                    this._currentActiveButton = -1;
                    this._previousPosition = null;
                    if (!noPreventDefault) {
                        evt.preventDefault();
                    }
                    this._activePointerId = -1;
                }
                else if (p.type === PointerEventTypes.POINTERMOVE && (this._activePointerId === evt.pointerId || !isTouch)) {
                    if (engine.isPointerLock && this._onMouseMove) {
                        this._onMouseMove(p.event);
                    }
                    else if (this._previousPosition) {
                        const handednessMultiplier = this.camera._calculateHandednessMultiplier();
                        const offsetX = (evt.clientX - this._previousPosition.x) * handednessMultiplier;
                        const offsetY = (evt.clientY - this._previousPosition.y) * handednessMultiplier;
                        if (this._allowCameraRotation) {
                            this.camera.cameraRotation.y += offsetX / this.angularSensibility;
                            this.camera.cameraRotation.x += offsetY / this.angularSensibility;
                        }
                        this.onPointerMovedObservable.notifyObservers({ offsetX: offsetX, offsetY: offsetY });
                        this._previousPosition = {
                            x: evt.clientX,
                            y: evt.clientY,
                        };
                        if (!noPreventDefault) {
                            evt.preventDefault();
                        }
                    }
                }
            };
        }
        this._onMouseMove = (evt) => {
            if (!engine.isPointerLock) {
                return;
            }
            const handednessMultiplier = this.camera._calculateHandednessMultiplier();
            this.camera.cameraRotation.y += (evt.movementX * handednessMultiplier) / this.angularSensibility;
            this.camera.cameraRotation.x += (evt.movementY * handednessMultiplier) / this.angularSensibility;
            this._previousPosition = null;
            if (!noPreventDefault) {
                evt.preventDefault();
            }
        };
        this._observer = this.camera
            .getScene()
            ._inputManager._addCameraPointerObserver(this._pointerInput, PointerEventTypes.POINTERDOWN | PointerEventTypes.POINTERUP | PointerEventTypes.POINTERMOVE);
        if (element) {
            this._contextMenuBind = (evt) => this.onContextMenu(evt);
            element.addEventListener("contextmenu", this._contextMenuBind, false); // TODO: We need to figure out how to handle this for Native
        }
    }
    /**
     * Called on JS contextmenu event.
     * Override this method to provide functionality.
     * @param evt the context menu event
     */
    onContextMenu(evt) {
        evt.preventDefault();
    }
    /**
     * Detach the current controls from the specified dom element.
     */
    detachControl() {
        if (this._observer) {
            this.camera.getScene()._inputManager._removeCameraPointerObserver(this._observer);
            if (this._contextMenuBind) {
                const engine = this.camera.getEngine();
                const element = engine.getInputElement();
                if (element) {
                    element.removeEventListener("contextmenu", this._contextMenuBind);
                }
            }
            if (this.onPointerMovedObservable) {
                this.onPointerMovedObservable.clear();
            }
            this._observer = null;
            this._onMouseMove = null;
            this._previousPosition = null;
        }
        this._activePointerId = -1;
        this._currentActiveButton = -1;
    }
    /**
     * Gets the class name of the current input.
     * @returns the class name
     */
    getClassName() {
        return "FreeCameraMouseInput";
    }
    /**
     * Get the friendly name associated with the input class.
     * @returns the input friendly name
     */
    getSimpleName() {
        return "mouse";
    }
}
__decorate([
    serialize()
], FreeCameraMouseInput.prototype, "buttons", void 0);
__decorate([
    serialize()
], FreeCameraMouseInput.prototype, "angularSensibility", void 0);
CameraInputTypes["FreeCameraMouseInput"] = FreeCameraMouseInput;

/**
 * Base class for mouse wheel input..
 * See FollowCameraMouseWheelInput in src/Cameras/Inputs/freeCameraMouseWheelInput.ts
 * for example usage.
 */
class BaseCameraMouseWheelInput {
    constructor() {
        /**
         * How fast is the camera moves in relation to X axis mouseWheel events.
         * Use negative value to reverse direction.
         */
        this.wheelPrecisionX = 3.0;
        /**
         * How fast is the camera moves in relation to Y axis mouseWheel events.
         * Use negative value to reverse direction.
         */
        this.wheelPrecisionY = 3.0;
        /**
         * How fast is the camera moves in relation to Z axis mouseWheel events.
         * Use negative value to reverse direction.
         */
        this.wheelPrecisionZ = 3.0;
        /**
         * Observable for when a mouse wheel move event occurs.
         */
        this.onChangedObservable = new Observable();
        /**
         * Incremental value of multiple mouse wheel movements of the X axis.
         * Should be zero-ed when read.
         */
        this._wheelDeltaX = 0;
        /**
         * Incremental value of multiple mouse wheel movements of the Y axis.
         * Should be zero-ed when read.
         */
        this._wheelDeltaY = 0;
        /**
         * Incremental value of multiple mouse wheel movements of the Z axis.
         * Should be zero-ed when read.
         */
        this._wheelDeltaZ = 0;
        /**
         * Firefox uses a different scheme to report scroll distances to other
         * browsers. Rather than use complicated methods to calculate the exact
         * multiple we need to apply, let's just cheat and use a constant.
         * https://developer.mozilla.org/en-US/docs/Web/API/WheelEvent/deltaMode
         * https://stackoverflow.com/questions/20110224/what-is-the-height-of-a-line-in-a-wheel-event-deltamode-dom-delta-line
         */
        this._ffMultiplier = 12;
        /**
         * Different event attributes for wheel data fall into a few set ranges.
         * Some relevant but dated date here:
         * https://stackoverflow.com/questions/5527601/normalizing-mousewheel-speed-across-browsers
         */
        this._normalize = 120;
    }
    /**
     * Attach the input controls to a specific dom element to get the input from.
     * @param noPreventDefault Defines whether event caught by the controls
     *   should call preventdefault().
     *   (https://developer.mozilla.org/en-US/docs/Web/API/Event/preventDefault)
     */
    attachControl(noPreventDefault) {
        noPreventDefault = Tools.BackCompatCameraNoPreventDefault(arguments);
        this._wheel = (pointer) => {
            // sanity check - this should be a PointerWheel event.
            if (pointer.type !== PointerEventTypes.POINTERWHEEL) {
                return;
            }
            const event = pointer.event;
            const platformScale = event.deltaMode === EventConstants.DOM_DELTA_LINE ? this._ffMultiplier : 1; // If this happens to be set to DOM_DELTA_LINE, adjust accordingly
            this._wheelDeltaX += (this.wheelPrecisionX * platformScale * event.deltaX) / this._normalize;
            this._wheelDeltaY -= (this.wheelPrecisionY * platformScale * event.deltaY) / this._normalize;
            this._wheelDeltaZ += (this.wheelPrecisionZ * platformScale * event.deltaZ) / this._normalize;
            if (event.preventDefault) {
                if (!noPreventDefault) {
                    event.preventDefault();
                }
            }
        };
        this._observer = this.camera.getScene()._inputManager._addCameraPointerObserver(this._wheel, PointerEventTypes.POINTERWHEEL);
    }
    /**
     * Detach the current controls from the specified dom element.
     */
    detachControl() {
        if (this._observer) {
            this.camera.getScene()._inputManager._removeCameraPointerObserver(this._observer);
            this._observer = null;
            this._wheel = null;
        }
        if (this.onChangedObservable) {
            this.onChangedObservable.clear();
        }
    }
    /**
     * Called for each rendered frame.
     */
    checkInputs() {
        this.onChangedObservable.notifyObservers({
            wheelDeltaX: this._wheelDeltaX,
            wheelDeltaY: this._wheelDeltaY,
            wheelDeltaZ: this._wheelDeltaZ,
        });
        // Clear deltas.
        this._wheelDeltaX = 0;
        this._wheelDeltaY = 0;
        this._wheelDeltaZ = 0;
    }
    /**
     * Gets the class name of the current input.
     * @returns the class name
     */
    getClassName() {
        return "BaseCameraMouseWheelInput";
    }
    /**
     * Get the friendly name associated with the input class.
     * @returns the input friendly name
     */
    getSimpleName() {
        return "mousewheel";
    }
}
__decorate([
    serialize()
], BaseCameraMouseWheelInput.prototype, "wheelPrecisionX", void 0);
__decorate([
    serialize()
], BaseCameraMouseWheelInput.prototype, "wheelPrecisionY", void 0);
__decorate([
    serialize()
], BaseCameraMouseWheelInput.prototype, "wheelPrecisionZ", void 0);

// eslint-disable-next-line @typescript-eslint/naming-convention
var _CameraProperty;
(function (_CameraProperty) {
    _CameraProperty[_CameraProperty["MoveRelative"] = 0] = "MoveRelative";
    _CameraProperty[_CameraProperty["RotateRelative"] = 1] = "RotateRelative";
    _CameraProperty[_CameraProperty["MoveScene"] = 2] = "MoveScene";
})(_CameraProperty || (_CameraProperty = {}));
/**
 * Manage the mouse wheel inputs to control a free camera.
 * @see https://doc.babylonjs.com/features/featuresDeepDive/cameras/customizingCameraInputs
 */
class FreeCameraMouseWheelInput extends BaseCameraMouseWheelInput {
    constructor() {
        super(...arguments);
        this._moveRelative = Vector3.Zero();
        this._rotateRelative = Vector3.Zero();
        this._moveScene = Vector3.Zero();
        /**
         * These are set to the desired default behaviour.
         */
        this._wheelXAction = _CameraProperty.MoveRelative;
        this._wheelXActionCoordinate = 0 /* Coordinate.X */;
        this._wheelYAction = _CameraProperty.MoveRelative;
        this._wheelYActionCoordinate = 2 /* Coordinate.Z */;
        this._wheelZAction = null;
        this._wheelZActionCoordinate = null;
    }
    /**
     * Gets the class name of the current input.
     * @returns the class name
     */
    getClassName() {
        return "FreeCameraMouseWheelInput";
    }
    /**
     * Set which movement axis (relative to camera's orientation) the mouse
     * wheel's X axis controls.
     * @param axis The axis to be moved. Set null to clear.
     */
    set wheelXMoveRelative(axis) {
        if (axis === null && this._wheelXAction !== _CameraProperty.MoveRelative) {
            // Attempting to clear different _wheelXAction.
            return;
        }
        this._wheelXAction = _CameraProperty.MoveRelative;
        this._wheelXActionCoordinate = axis;
    }
    /**
     * Get the configured movement axis (relative to camera's orientation) the
     * mouse wheel's X axis controls.
     * @returns The configured axis or null if none.
     */
    get wheelXMoveRelative() {
        if (this._wheelXAction !== _CameraProperty.MoveRelative) {
            return null;
        }
        return this._wheelXActionCoordinate;
    }
    /**
     * Set which movement axis (relative to camera's orientation) the mouse
     * wheel's Y axis controls.
     * @param axis The axis to be moved. Set null to clear.
     */
    set wheelYMoveRelative(axis) {
        if (axis === null && this._wheelYAction !== _CameraProperty.MoveRelative) {
            // Attempting to clear different _wheelYAction.
            return;
        }
        this._wheelYAction = _CameraProperty.MoveRelative;
        this._wheelYActionCoordinate = axis;
    }
    /**
     * Get the configured movement axis (relative to camera's orientation) the
     * mouse wheel's Y axis controls.
     * @returns The configured axis or null if none.
     */
    get wheelYMoveRelative() {
        if (this._wheelYAction !== _CameraProperty.MoveRelative) {
            return null;
        }
        return this._wheelYActionCoordinate;
    }
    /**
     * Set which movement axis (relative to camera's orientation) the mouse
     * wheel's Z axis controls.
     * @param axis The axis to be moved. Set null to clear.
     */
    set wheelZMoveRelative(axis) {
        if (axis === null && this._wheelZAction !== _CameraProperty.MoveRelative) {
            // Attempting to clear different _wheelZAction.
            return;
        }
        this._wheelZAction = _CameraProperty.MoveRelative;
        this._wheelZActionCoordinate = axis;
    }
    /**
     * Get the configured movement axis (relative to camera's orientation) the
     * mouse wheel's Z axis controls.
     * @returns The configured axis or null if none.
     */
    get wheelZMoveRelative() {
        if (this._wheelZAction !== _CameraProperty.MoveRelative) {
            return null;
        }
        return this._wheelZActionCoordinate;
    }
    /**
     * Set which rotation axis (relative to camera's orientation) the mouse
     * wheel's X axis controls.
     * @param axis The axis to be moved. Set null to clear.
     */
    set wheelXRotateRelative(axis) {
        if (axis === null && this._wheelXAction !== _CameraProperty.RotateRelative) {
            // Attempting to clear different _wheelXAction.
            return;
        }
        this._wheelXAction = _CameraProperty.RotateRelative;
        this._wheelXActionCoordinate = axis;
    }
    /**
     * Get the configured rotation axis (relative to camera's orientation) the
     * mouse wheel's X axis controls.
     * @returns The configured axis or null if none.
     */
    get wheelXRotateRelative() {
        if (this._wheelXAction !== _CameraProperty.RotateRelative) {
            return null;
        }
        return this._wheelXActionCoordinate;
    }
    /**
     * Set which rotation axis (relative to camera's orientation) the mouse
     * wheel's Y axis controls.
     * @param axis The axis to be moved. Set null to clear.
     */
    set wheelYRotateRelative(axis) {
        if (axis === null && this._wheelYAction !== _CameraProperty.RotateRelative) {
            // Attempting to clear different _wheelYAction.
            return;
        }
        this._wheelYAction = _CameraProperty.RotateRelative;
        this._wheelYActionCoordinate = axis;
    }
    /**
     * Get the configured rotation axis (relative to camera's orientation) the
     * mouse wheel's Y axis controls.
     * @returns The configured axis or null if none.
     */
    get wheelYRotateRelative() {
        if (this._wheelYAction !== _CameraProperty.RotateRelative) {
            return null;
        }
        return this._wheelYActionCoordinate;
    }
    /**
     * Set which rotation axis (relative to camera's orientation) the mouse
     * wheel's Z axis controls.
     * @param axis The axis to be moved. Set null to clear.
     */
    set wheelZRotateRelative(axis) {
        if (axis === null && this._wheelZAction !== _CameraProperty.RotateRelative) {
            // Attempting to clear different _wheelZAction.
            return;
        }
        this._wheelZAction = _CameraProperty.RotateRelative;
        this._wheelZActionCoordinate = axis;
    }
    /**
     * Get the configured rotation axis (relative to camera's orientation) the
     * mouse wheel's Z axis controls.
     * @returns The configured axis or null if none.
     */
    get wheelZRotateRelative() {
        if (this._wheelZAction !== _CameraProperty.RotateRelative) {
            return null;
        }
        return this._wheelZActionCoordinate;
    }
    /**
     * Set which movement axis (relative to the scene) the mouse wheel's X axis
     * controls.
     * @param axis The axis to be moved. Set null to clear.
     */
    set wheelXMoveScene(axis) {
        if (axis === null && this._wheelXAction !== _CameraProperty.MoveScene) {
            // Attempting to clear different _wheelXAction.
            return;
        }
        this._wheelXAction = _CameraProperty.MoveScene;
        this._wheelXActionCoordinate = axis;
    }
    /**
     * Get the configured movement axis (relative to the scene) the mouse wheel's
     * X axis controls.
     * @returns The configured axis or null if none.
     */
    get wheelXMoveScene() {
        if (this._wheelXAction !== _CameraProperty.MoveScene) {
            return null;
        }
        return this._wheelXActionCoordinate;
    }
    /**
     * Set which movement axis (relative to the scene) the mouse wheel's Y axis
     * controls.
     * @param axis The axis to be moved. Set null to clear.
     */
    set wheelYMoveScene(axis) {
        if (axis === null && this._wheelYAction !== _CameraProperty.MoveScene) {
            // Attempting to clear different _wheelYAction.
            return;
        }
        this._wheelYAction = _CameraProperty.MoveScene;
        this._wheelYActionCoordinate = axis;
    }
    /**
     * Get the configured movement axis (relative to the scene) the mouse wheel's
     * Y axis controls.
     * @returns The configured axis or null if none.
     */
    get wheelYMoveScene() {
        if (this._wheelYAction !== _CameraProperty.MoveScene) {
            return null;
        }
        return this._wheelYActionCoordinate;
    }
    /**
     * Set which movement axis (relative to the scene) the mouse wheel's Z axis
     * controls.
     * @param axis The axis to be moved. Set null to clear.
     */
    set wheelZMoveScene(axis) {
        if (axis === null && this._wheelZAction !== _CameraProperty.MoveScene) {
            // Attempting to clear different _wheelZAction.
            return;
        }
        this._wheelZAction = _CameraProperty.MoveScene;
        this._wheelZActionCoordinate = axis;
    }
    /**
     * Get the configured movement axis (relative to the scene) the mouse wheel's
     * Z axis controls.
     * @returns The configured axis or null if none.
     */
    get wheelZMoveScene() {
        if (this._wheelZAction !== _CameraProperty.MoveScene) {
            return null;
        }
        return this._wheelZActionCoordinate;
    }
    /**
     * Called for each rendered frame.
     */
    checkInputs() {
        if (this._wheelDeltaX === 0 && this._wheelDeltaY === 0 && this._wheelDeltaZ == 0) {
            return;
        }
        // Clear the camera properties that we might be updating.
        this._moveRelative.setAll(0);
        this._rotateRelative.setAll(0);
        this._moveScene.setAll(0);
        // Set the camera properties that are to be updated.
        this._updateCamera();
        if (this.camera.getScene().useRightHandedSystem) {
            // TODO: Does this need done for worldUpdate too?
            this._moveRelative.z *= -1;
        }
        // Convert updates relative to camera to world position update.
        const cameraTransformMatrix = Matrix.Zero();
        this.camera.getViewMatrix().invertToRef(cameraTransformMatrix);
        const transformedDirection = Vector3.Zero();
        Vector3.TransformNormalToRef(this._moveRelative, cameraTransformMatrix, transformedDirection);
        // Apply updates to camera position.
        this.camera.cameraRotation.x += this._rotateRelative.x / 200;
        this.camera.cameraRotation.y += this._rotateRelative.y / 200;
        this.camera.cameraDirection.addInPlace(transformedDirection);
        this.camera.cameraDirection.addInPlace(this._moveScene);
        // Call the base class implementation to handle observers and do cleanup.
        super.checkInputs();
    }
    /**
     * Update the camera according to any configured properties for the 3
     * mouse-wheel axis.
     */
    _updateCamera() {
        // Do the camera updates for each of the 3 touch-wheel axis.
        this._updateCameraProperty(this._wheelDeltaX, this._wheelXAction, this._wheelXActionCoordinate);
        this._updateCameraProperty(this._wheelDeltaY, this._wheelYAction, this._wheelYActionCoordinate);
        this._updateCameraProperty(this._wheelDeltaZ, this._wheelZAction, this._wheelZActionCoordinate);
    }
    /**
     * Update one property of the camera.
     * @param value
     * @param cameraProperty
     * @param coordinate
     */
    _updateCameraProperty(
    /* Mouse-wheel delta. */
    value, 
    /* Camera property to be changed. */
    cameraProperty, 
    /* Axis of Camera property to be changed. */
    coordinate) {
        if (value === 0) {
            // Mouse wheel has not moved.
            return;
        }
        if (cameraProperty === null || coordinate === null) {
            // Mouse wheel axis not configured.
            return;
        }
        let action = null;
        switch (cameraProperty) {
            case _CameraProperty.MoveRelative:
                action = this._moveRelative;
                break;
            case _CameraProperty.RotateRelative:
                action = this._rotateRelative;
                break;
            case _CameraProperty.MoveScene:
                action = this._moveScene;
                break;
        }
        switch (coordinate) {
            case 0 /* Coordinate.X */:
                action.set(value, 0, 0);
                break;
            case 1 /* Coordinate.Y */:
                action.set(0, value, 0);
                break;
            case 2 /* Coordinate.Z */:
                action.set(0, 0, value);
                break;
        }
    }
}
__decorate([
    serialize()
], FreeCameraMouseWheelInput.prototype, "wheelXMoveRelative", null);
__decorate([
    serialize()
], FreeCameraMouseWheelInput.prototype, "wheelYMoveRelative", null);
__decorate([
    serialize()
], FreeCameraMouseWheelInput.prototype, "wheelZMoveRelative", null);
__decorate([
    serialize()
], FreeCameraMouseWheelInput.prototype, "wheelXRotateRelative", null);
__decorate([
    serialize()
], FreeCameraMouseWheelInput.prototype, "wheelYRotateRelative", null);
__decorate([
    serialize()
], FreeCameraMouseWheelInput.prototype, "wheelZRotateRelative", null);
__decorate([
    serialize()
], FreeCameraMouseWheelInput.prototype, "wheelXMoveScene", null);
__decorate([
    serialize()
], FreeCameraMouseWheelInput.prototype, "wheelYMoveScene", null);
__decorate([
    serialize()
], FreeCameraMouseWheelInput.prototype, "wheelZMoveScene", null);
CameraInputTypes["FreeCameraMouseWheelInput"] = FreeCameraMouseWheelInput;

/**
 * Manage the touch inputs to control the movement of a free camera.
 * @see https://doc.babylonjs.com/features/featuresDeepDive/cameras/customizingCameraInputs
 */
class FreeCameraTouchInput {
    /**
     * Manage the touch inputs to control the movement of a free camera.
     * @see https://doc.babylonjs.com/features/featuresDeepDive/cameras/customizingCameraInputs
     * @param allowMouse Defines if mouse events can be treated as touch events
     */
    constructor(
    /**
     * [false] Define if mouse events can be treated as touch events
     */
    allowMouse = false) {
        this.allowMouse = allowMouse;
        /**
         * Defines the touch sensibility for rotation.
         * The lower the faster.
         */
        this.touchAngularSensibility = 200000.0;
        /**
         * Defines the touch sensibility for move.
         * The lower the faster.
         */
        this.touchMoveSensibility = 250.0;
        /**
         * Swap touch actions so that one touch is used for rotation and multiple for movement
         */
        this.singleFingerRotate = false;
        this._offsetX = null;
        this._offsetY = null;
        this._pointerPressed = new Array();
        this._isSafari = Tools.IsSafari();
    }
    /**
     * Attach the input controls to a specific dom element to get the input from.
     * @param noPreventDefault Defines whether event caught by the controls should call preventdefault() (https://developer.mozilla.org/en-US/docs/Web/API/Event/preventDefault)
     */
    attachControl(noPreventDefault) {
        noPreventDefault = Tools.BackCompatCameraNoPreventDefault(arguments);
        let previousPosition = null;
        if (this._pointerInput === undefined) {
            this._onLostFocus = () => {
                this._offsetX = null;
                this._offsetY = null;
            };
            this._pointerInput = (p) => {
                const evt = p.event;
                const isMouseEvent = evt.pointerType === "mouse" || (this._isSafari && typeof evt.pointerType === "undefined");
                if (!this.allowMouse && isMouseEvent) {
                    return;
                }
                if (p.type === PointerEventTypes.POINTERDOWN) {
                    if (!noPreventDefault) {
                        evt.preventDefault();
                    }
                    this._pointerPressed.push(evt.pointerId);
                    if (this._pointerPressed.length !== 1) {
                        return;
                    }
                    previousPosition = {
                        x: evt.clientX,
                        y: evt.clientY,
                    };
                }
                else if (p.type === PointerEventTypes.POINTERUP) {
                    if (!noPreventDefault) {
                        evt.preventDefault();
                    }
                    const index = this._pointerPressed.indexOf(evt.pointerId);
                    if (index === -1) {
                        return;
                    }
                    this._pointerPressed.splice(index, 1);
                    if (index != 0) {
                        return;
                    }
                    previousPosition = null;
                    this._offsetX = null;
                    this._offsetY = null;
                }
                else if (p.type === PointerEventTypes.POINTERMOVE) {
                    if (!noPreventDefault) {
                        evt.preventDefault();
                    }
                    if (!previousPosition) {
                        return;
                    }
                    const index = this._pointerPressed.indexOf(evt.pointerId);
                    if (index != 0) {
                        return;
                    }
                    this._offsetX = evt.clientX - previousPosition.x;
                    this._offsetY = -(evt.clientY - previousPosition.y);
                }
            };
        }
        this._observer = this.camera
            .getScene()
            ._inputManager._addCameraPointerObserver(this._pointerInput, PointerEventTypes.POINTERDOWN | PointerEventTypes.POINTERUP | PointerEventTypes.POINTERMOVE);
        if (this._onLostFocus) {
            const engine = this.camera.getEngine();
            const element = engine.getInputElement();
            if (element) {
                element.addEventListener("blur", this._onLostFocus);
            }
        }
    }
    /**
     * Detach the current controls from the specified dom element.
     */
    detachControl() {
        if (this._pointerInput) {
            if (this._observer) {
                this.camera.getScene()._inputManager._removeCameraPointerObserver(this._observer);
                this._observer = null;
            }
            if (this._onLostFocus) {
                const engine = this.camera.getEngine();
                const element = engine.getInputElement();
                if (element) {
                    element.removeEventListener("blur", this._onLostFocus);
                }
                this._onLostFocus = null;
            }
            this._pointerPressed.length = 0;
            this._offsetX = null;
            this._offsetY = null;
        }
    }
    /**
     * Update the current camera state depending on the inputs that have been used this frame.
     * This is a dynamically created lambda to avoid the performance penalty of looping for inputs in the render loop.
     */
    checkInputs() {
        if (this._offsetX === null || this._offsetY === null) {
            return;
        }
        if (this._offsetX === 0 && this._offsetY === 0) {
            return;
        }
        const camera = this.camera;
        const handednessMultiplier = camera._calculateHandednessMultiplier();
        camera.cameraRotation.y = (this._offsetX * handednessMultiplier) / this.touchAngularSensibility;
        const rotateCamera = (this.singleFingerRotate && this._pointerPressed.length === 1) || (!this.singleFingerRotate && this._pointerPressed.length > 1);
        if (rotateCamera) {
            camera.cameraRotation.x = -(this._offsetY * handednessMultiplier) / this.touchAngularSensibility;
        }
        else {
            const speed = camera._computeLocalCameraSpeed();
            const direction = new Vector3(0, 0, this.touchMoveSensibility !== 0 ? (speed * this._offsetY) / this.touchMoveSensibility : 0);
            Matrix.RotationYawPitchRollToRef(camera.rotation.y, camera.rotation.x, 0, camera._cameraRotationMatrix);
            camera.cameraDirection.addInPlace(Vector3.TransformCoordinates(direction, camera._cameraRotationMatrix));
        }
    }
    /**
     * Gets the class name of the current input.
     * @returns the class name
     */
    getClassName() {
        return "FreeCameraTouchInput";
    }
    /**
     * Get the friendly name associated with the input class.
     * @returns the input friendly name
     */
    getSimpleName() {
        return "touch";
    }
}
__decorate([
    serialize()
], FreeCameraTouchInput.prototype, "touchAngularSensibility", void 0);
__decorate([
    serialize()
], FreeCameraTouchInput.prototype, "touchMoveSensibility", void 0);
CameraInputTypes["FreeCameraTouchInput"] = FreeCameraTouchInput;

/**
 * Default Inputs manager for the FreeCamera.
 * It groups all the default supported inputs for ease of use.
 * @see https://doc.babylonjs.com/features/featuresDeepDive/cameras/customizingCameraInputs
 */
class FreeCameraInputsManager extends CameraInputsManager {
    /**
     * Instantiates a new FreeCameraInputsManager.
     * @param camera Defines the camera the inputs belong to
     */
    constructor(camera) {
        super(camera);
        /**
         * @internal
         */
        this._mouseInput = null;
        /**
         * @internal
         */
        this._mouseWheelInput = null;
    }
    /**
     * Add keyboard input support to the input manager.
     * @returns the current input manager
     */
    addKeyboard() {
        this.add(new FreeCameraKeyboardMoveInput());
        return this;
    }
    /**
     * Add mouse input support to the input manager.
     * @param touchEnabled if the FreeCameraMouseInput should support touch (default: true)
     * @returns the current input manager
     */
    addMouse(touchEnabled = true) {
        if (!this._mouseInput) {
            this._mouseInput = new FreeCameraMouseInput(touchEnabled);
            this.add(this._mouseInput);
        }
        return this;
    }
    /**
     * Removes the mouse input support from the manager
     * @returns the current input manager
     */
    removeMouse() {
        if (this._mouseInput) {
            this.remove(this._mouseInput);
        }
        return this;
    }
    /**
     * Add mouse wheel input support to the input manager.
     * @returns the current input manager
     */
    addMouseWheel() {
        if (!this._mouseWheelInput) {
            this._mouseWheelInput = new FreeCameraMouseWheelInput();
            this.add(this._mouseWheelInput);
        }
        return this;
    }
    /**
     * Removes the mouse wheel input support from the manager
     * @returns the current input manager
     */
    removeMouseWheel() {
        if (this._mouseWheelInput) {
            this.remove(this._mouseWheelInput);
        }
        return this;
    }
    /**
     * Add touch input support to the input manager.
     * @returns the current input manager
     */
    addTouch() {
        this.add(new FreeCameraTouchInput());
        return this;
    }
    /**
     * Remove all attached input methods from a camera
     */
    clear() {
        super.clear();
        this._mouseInput = null;
    }
}

/**
 * This represents a free type of camera. It can be useful in First Person Shooter game for instance.
 * Please consider using the new UniversalCamera instead as it adds more functionality like the gamepad.
 * @see https://doc.babylonjs.com/features/featuresDeepDive/cameras/camera_introduction#universal-camera
 */
class FreeCamera extends TargetCamera {
    /**
     * Gets the input sensibility for a mouse input. (default is 2000.0)
     * Higher values reduce sensitivity.
     */
    get angularSensibility() {
        const mouse = this.inputs.attached["mouse"];
        if (mouse) {
            return mouse.angularSensibility;
        }
        return 0;
    }
    /**
     * Sets the input sensibility for a mouse input. (default is 2000.0)
     * Higher values reduce sensitivity.
     */
    set angularSensibility(value) {
        const mouse = this.inputs.attached["mouse"];
        if (mouse) {
            mouse.angularSensibility = value;
        }
    }
    /**
     * Gets or Set the list of keyboard keys used to control the forward move of the camera.
     */
    get keysUp() {
        const keyboard = this.inputs.attached["keyboard"];
        if (keyboard) {
            return keyboard.keysUp;
        }
        return [];
    }
    set keysUp(value) {
        const keyboard = this.inputs.attached["keyboard"];
        if (keyboard) {
            keyboard.keysUp = value;
        }
    }
    /**
     * Gets or Set the list of keyboard keys used to control the upward move of the camera.
     */
    get keysUpward() {
        const keyboard = this.inputs.attached["keyboard"];
        if (keyboard) {
            return keyboard.keysUpward;
        }
        return [];
    }
    set keysUpward(value) {
        const keyboard = this.inputs.attached["keyboard"];
        if (keyboard) {
            keyboard.keysUpward = value;
        }
    }
    /**
     * Gets or Set the list of keyboard keys used to control the backward move of the camera.
     */
    get keysDown() {
        const keyboard = this.inputs.attached["keyboard"];
        if (keyboard) {
            return keyboard.keysDown;
        }
        return [];
    }
    set keysDown(value) {
        const keyboard = this.inputs.attached["keyboard"];
        if (keyboard) {
            keyboard.keysDown = value;
        }
    }
    /**
     * Gets or Set the list of keyboard keys used to control the downward move of the camera.
     */
    get keysDownward() {
        const keyboard = this.inputs.attached["keyboard"];
        if (keyboard) {
            return keyboard.keysDownward;
        }
        return [];
    }
    set keysDownward(value) {
        const keyboard = this.inputs.attached["keyboard"];
        if (keyboard) {
            keyboard.keysDownward = value;
        }
    }
    /**
     * Gets or Set the list of keyboard keys used to control the left strafe move of the camera.
     */
    get keysLeft() {
        const keyboard = this.inputs.attached["keyboard"];
        if (keyboard) {
            return keyboard.keysLeft;
        }
        return [];
    }
    set keysLeft(value) {
        const keyboard = this.inputs.attached["keyboard"];
        if (keyboard) {
            keyboard.keysLeft = value;
        }
    }
    /**
     * Gets or Set the list of keyboard keys used to control the right strafe move of the camera.
     */
    get keysRight() {
        const keyboard = this.inputs.attached["keyboard"];
        if (keyboard) {
            return keyboard.keysRight;
        }
        return [];
    }
    set keysRight(value) {
        const keyboard = this.inputs.attached["keyboard"];
        if (keyboard) {
            keyboard.keysRight = value;
        }
    }
    /**
     * Gets or Set the list of keyboard keys used to control the left rotation move of the camera.
     */
    get keysRotateLeft() {
        const keyboard = this.inputs.attached["keyboard"];
        if (keyboard) {
            return keyboard.keysRotateLeft;
        }
        return [];
    }
    set keysRotateLeft(value) {
        const keyboard = this.inputs.attached["keyboard"];
        if (keyboard) {
            keyboard.keysRotateLeft = value;
        }
    }
    /**
     * Gets or Set the list of keyboard keys used to control the right rotation move of the camera.
     */
    get keysRotateRight() {
        const keyboard = this.inputs.attached["keyboard"];
        if (keyboard) {
            return keyboard.keysRotateRight;
        }
        return [];
    }
    set keysRotateRight(value) {
        const keyboard = this.inputs.attached["keyboard"];
        if (keyboard) {
            keyboard.keysRotateRight = value;
        }
    }
    /**
     * Gets or Set the list of keyboard keys used to control the up rotation move of the camera.
     */
    get keysRotateUp() {
        const keyboard = this.inputs.attached["keyboard"];
        if (keyboard) {
            return keyboard.keysRotateUp;
        }
        return [];
    }
    set keysRotateUp(value) {
        const keyboard = this.inputs.attached["keyboard"];
        if (keyboard) {
            keyboard.keysRotateUp = value;
        }
    }
    /**
     * Gets or Set the list of keyboard keys used to control the down rotation move of the camera.
     */
    get keysRotateDown() {
        const keyboard = this.inputs.attached["keyboard"];
        if (keyboard) {
            return keyboard.keysRotateDown;
        }
        return [];
    }
    set keysRotateDown(value) {
        const keyboard = this.inputs.attached["keyboard"];
        if (keyboard) {
            keyboard.keysRotateDown = value;
        }
    }
    /**
     * Instantiates a Free Camera.
     * This represents a free type of camera. It can be useful in First Person Shooter game for instance.
     * Please consider using the new UniversalCamera instead as it adds more functionality like touch to this camera.
     * @see https://doc.babylonjs.com/features/featuresDeepDive/cameras/camera_introduction#universal-camera
     * @param name Define the name of the camera in the scene
     * @param position Define the start position of the camera in the scene
     * @param scene Define the scene the camera belongs to
     * @param setActiveOnSceneIfNoneActive Defines whether the camera should be marked as active if not other active cameras have been defined
     */
    constructor(name, position, scene, setActiveOnSceneIfNoneActive = true) {
        super(name, position, scene, setActiveOnSceneIfNoneActive);
        /**
         * Define the collision ellipsoid of the camera.
         * This is helpful to simulate a camera body like the player body around the camera
         * @see https://doc.babylonjs.com/features/featuresDeepDive/cameras/camera_collisions#arcrotatecamera
         */
        this.ellipsoid = new Vector3(0.5, 1, 0.5);
        /**
         * Define an offset for the position of the ellipsoid around the camera.
         * This can be helpful to determine the center of the body near the gravity center of the body
         * instead of its head.
         */
        this.ellipsoidOffset = new Vector3(0, 0, 0);
        /**
         * Enable or disable collisions of the camera with the rest of the scene objects.
         */
        this.checkCollisions = false;
        /**
         * Enable or disable gravity on the camera.
         */
        this.applyGravity = false;
        this._needMoveForGravity = false;
        this._oldPosition = Vector3.Zero();
        this._diffPosition = Vector3.Zero();
        this._newPosition = Vector3.Zero();
        // Collisions
        this._collisionMask = -1;
        this._onCollisionPositionChange = (collisionId, newPosition, collidedMesh = null) => {
            this._newPosition.copyFrom(newPosition);
            this._newPosition.subtractToRef(this._oldPosition, this._diffPosition);
            if (this._diffPosition.length() > AbstractEngine.CollisionsEpsilon) {
                this.position.addToRef(this._diffPosition, this._deferredPositionUpdate);
                if (!this._deferOnly) {
                    this.position.copyFrom(this._deferredPositionUpdate);
                }
                else {
                    this._deferredUpdated = true;
                }
                // call onCollide, if defined. Note that in case of deferred update, the actual position change might happen in the next frame.
                if (this.onCollide && collidedMesh) {
                    this.onCollide(collidedMesh);
                }
            }
        };
        this.inputs = new FreeCameraInputsManager(this);
        this.inputs.addKeyboard().addMouse();
    }
    /**
     * Attached controls to the current camera.
     * @param ignored defines an ignored parameter kept for backward compatibility.
     * @param noPreventDefault Defines whether event caught by the controls should call preventdefault() (https://developer.mozilla.org/en-US/docs/Web/API/Event/preventDefault)
     */
    attachControl(ignored, noPreventDefault) {
        noPreventDefault = Tools.BackCompatCameraNoPreventDefault(arguments);
        this.inputs.attachElement(noPreventDefault);
    }
    /**
     * Detach the current controls from the specified dom element.
     */
    detachControl() {
        this.inputs.detachElement();
        this.cameraDirection = new Vector3(0, 0, 0);
        this.cameraRotation = new Vector2(0, 0);
    }
    /**
     * Define a collision mask to limit the list of object the camera can collide with
     */
    get collisionMask() {
        return this._collisionMask;
    }
    set collisionMask(mask) {
        this._collisionMask = !isNaN(mask) ? mask : -1;
    }
    /**
     * @internal
     */
    _collideWithWorld(displacement) {
        let globalPosition;
        if (this.parent) {
            globalPosition = Vector3.TransformCoordinates(this.position, this.parent.getWorldMatrix());
        }
        else {
            globalPosition = this.position;
        }
        globalPosition.subtractFromFloatsToRef(0, this.ellipsoid.y, 0, this._oldPosition);
        this._oldPosition.addInPlace(this.ellipsoidOffset);
        const coordinator = this.getScene().collisionCoordinator;
        if (!this._collider) {
            this._collider = coordinator.createCollider();
        }
        this._collider._radius = this.ellipsoid;
        this._collider.collisionMask = this._collisionMask;
        //no need for clone, as long as gravity is not on.
        let actualDisplacement = displacement;
        //add gravity to the direction to prevent the dual-collision checking
        if (this.applyGravity) {
            //this prevents mending with cameraDirection, a global variable of the free camera class.
            actualDisplacement = displacement.add(this.getScene().gravity);
        }
        coordinator.getNewPosition(this._oldPosition, actualDisplacement, this._collider, 3, null, this._onCollisionPositionChange, this.uniqueId);
    }
    /** @internal */
    _checkInputs() {
        if (!this._localDirection) {
            this._localDirection = Vector3.Zero();
            this._transformedDirection = Vector3.Zero();
        }
        this.inputs.checkInputs();
        super._checkInputs();
    }
    /**
     * Enable movement without a user input. This allows gravity to always be applied.
     */
    set needMoveForGravity(value) {
        this._needMoveForGravity = value;
    }
    /**
     * When true, gravity is applied whether there is user input or not.
     */
    get needMoveForGravity() {
        return this._needMoveForGravity;
    }
    /** @internal */
    _decideIfNeedsToMove() {
        return this._needMoveForGravity || Math.abs(this.cameraDirection.x) > 0 || Math.abs(this.cameraDirection.y) > 0 || Math.abs(this.cameraDirection.z) > 0;
    }
    /** @internal */
    _updatePosition() {
        if (this.checkCollisions && this.getScene().collisionsEnabled) {
            this._collideWithWorld(this.cameraDirection);
        }
        else {
            super._updatePosition();
        }
    }
    /**
     * Destroy the camera and release the current resources hold by it.
     */
    dispose() {
        this.inputs.clear();
        super.dispose();
    }
    /**
     * Gets the current object class name.
     * @returns the class name
     */
    getClassName() {
        return "FreeCamera";
    }
}
__decorate([
    serializeAsVector3()
], FreeCamera.prototype, "ellipsoid", void 0);
__decorate([
    serializeAsVector3()
], FreeCamera.prototype, "ellipsoidOffset", void 0);
__decorate([
    serialize()
], FreeCamera.prototype, "checkCollisions", void 0);
__decorate([
    serialize()
], FreeCamera.prototype, "applyGravity", void 0);
// Register Class Name
RegisterClass("BABYLON.FreeCamera", FreeCamera);

/**
 * Defines a target to use with MorphTargetManager
 * @see https://doc.babylonjs.com/features/featuresDeepDive/mesh/morphTargets
 */
class MorphTarget {
    /**
     * Gets or sets the influence of this target (ie. its weight in the overall morphing)
     */
    get influence() {
        return this._influence;
    }
    set influence(influence) {
        if (this._influence === influence) {
            return;
        }
        const previous = this._influence;
        this._influence = influence;
        if (this.onInfluenceChanged.hasObservers()) {
            this.onInfluenceChanged.notifyObservers(previous === 0 || influence === 0);
        }
    }
    /**
     * Gets or sets the animation properties override
     */
    get animationPropertiesOverride() {
        if (!this._animationPropertiesOverride && this._scene) {
            return this._scene.animationPropertiesOverride;
        }
        return this._animationPropertiesOverride;
    }
    set animationPropertiesOverride(value) {
        this._animationPropertiesOverride = value;
    }
    /**
     * Creates a new MorphTarget
     * @param name defines the name of the target
     * @param influence defines the influence to use
     * @param scene defines the scene the morphtarget belongs to
     */
    constructor(
    /** defines the name of the target */
    name, influence = 0, scene = null) {
        this.name = name;
        /**
         * Gets or sets the list of animations
         */
        this.animations = [];
        this._positions = null;
        this._normals = null;
        this._tangents = null;
        this._uvs = null;
        this._uv2s = null;
        this._colors = null;
        this._uniqueId = 0;
        /**
         * Observable raised when the influence changes
         */
        this.onInfluenceChanged = new Observable();
        /** @internal */
        this._onDataLayoutChanged = new Observable();
        this._animationPropertiesOverride = null;
        this.id = name;
        this._scene = scene || EngineStore.LastCreatedScene;
        this.influence = influence;
        if (this._scene) {
            this._uniqueId = this._scene.getUniqueId();
        }
    }
    /**
     * Gets the unique ID of this manager
     */
    get uniqueId() {
        return this._uniqueId;
    }
    /**
     * Gets a boolean defining if the target contains position data
     */
    get hasPositions() {
        return !!this._positions;
    }
    /**
     * Gets a boolean defining if the target contains normal data
     */
    get hasNormals() {
        return !!this._normals;
    }
    /**
     * Gets a boolean defining if the target contains tangent data
     */
    get hasTangents() {
        return !!this._tangents;
    }
    /**
     * Gets a boolean defining if the target contains texture coordinates data
     */
    get hasUVs() {
        return !!this._uvs;
    }
    /**
     * Gets a boolean defining if the target contains texture coordinates 2 data
     */
    get hasUV2s() {
        return !!this._uv2s;
    }
    get hasColors() {
        return !!this._colors;
    }
    /**
     * Gets the number of vertices stored in this target
     */
    get vertexCount() {
        return this._positions
            ? this._positions.length / 3
            : this._normals
                ? this._normals.length / 3
                : this._tangents
                    ? this._tangents.length / 3
                    : this._uvs
                        ? this._uvs.length / 2
                        : this._uv2s
                            ? this._uv2s.length / 2
                            : this._colors
                                ? this._colors.length / 4
                                : 0;
    }
    /**
     * Affects position data to this target
     * @param data defines the position data to use
     */
    setPositions(data) {
        const hadPositions = this.hasPositions;
        this._positions = data;
        if (hadPositions !== this.hasPositions) {
            this._onDataLayoutChanged.notifyObservers(undefined);
        }
    }
    /**
     * Gets the position data stored in this target
     * @returns a FloatArray containing the position data (or null if not present)
     */
    getPositions() {
        return this._positions;
    }
    /**
     * Affects normal data to this target
     * @param data defines the normal data to use
     */
    setNormals(data) {
        const hadNormals = this.hasNormals;
        this._normals = data;
        if (hadNormals !== this.hasNormals) {
            this._onDataLayoutChanged.notifyObservers(undefined);
        }
    }
    /**
     * Gets the normal data stored in this target
     * @returns a FloatArray containing the normal data (or null if not present)
     */
    getNormals() {
        return this._normals;
    }
    /**
     * Affects tangent data to this target
     * @param data defines the tangent data to use
     */
    setTangents(data) {
        const hadTangents = this.hasTangents;
        this._tangents = data;
        if (hadTangents !== this.hasTangents) {
            this._onDataLayoutChanged.notifyObservers(undefined);
        }
    }
    /**
     * Gets the tangent data stored in this target
     * @returns a FloatArray containing the tangent data (or null if not present)
     */
    getTangents() {
        return this._tangents;
    }
    /**
     * Affects texture coordinates data to this target
     * @param data defines the texture coordinates data to use
     */
    setUVs(data) {
        const hadUVs = this.hasUVs;
        this._uvs = data;
        if (hadUVs !== this.hasUVs) {
            this._onDataLayoutChanged.notifyObservers(undefined);
        }
    }
    /**
     * Gets the texture coordinates data stored in this target
     * @returns a FloatArray containing the texture coordinates data (or null if not present)
     */
    getUVs() {
        return this._uvs;
    }
    /**
     * Affects texture coordinates 2 data to this target
     * @param data defines the texture coordinates 2 data to use
     */
    setUV2s(data) {
        const hadUV2s = this.hasUV2s;
        this._uv2s = data;
        if (hadUV2s !== this.hasUV2s) {
            this._onDataLayoutChanged.notifyObservers(undefined);
        }
    }
    /**
     * Gets the texture coordinates 2 data stored in this target
     * @returns a FloatArray containing the texture coordinates 2 data (or null if not present)
     */
    getUV2s() {
        return this._uv2s;
    }
    /**
     * Affects color data to this target
     * @param data defines the color data to use
     */
    setColors(data) {
        const hadColors = this.hasColors;
        this._colors = data;
        if (hadColors !== this.hasColors) {
            this._onDataLayoutChanged.notifyObservers(undefined);
        }
    }
    /**
     * Gets the color data stored in this target
     * @returns a FloatArray containing the color data (or null if not present)
     */
    getColors() {
        return this._colors;
    }
    /**
     * Clone the current target
     * @returns a new MorphTarget
     */
    clone() {
        const newOne = SerializationHelper.Clone(() => new MorphTarget(this.name, this.influence, this._scene), this);
        newOne._positions = this._positions;
        newOne._normals = this._normals;
        newOne._tangents = this._tangents;
        newOne._uvs = this._uvs;
        newOne._uv2s = this._uv2s;
        newOne._colors = this._colors;
        return newOne;
    }
    /**
     * Serializes the current target into a Serialization object
     * @returns the serialized object
     */
    serialize() {
        const serializationObject = {};
        serializationObject.name = this.name;
        serializationObject.influence = this.influence;
        serializationObject.positions = Array.prototype.slice.call(this.getPositions());
        if (this.id != null) {
            serializationObject.id = this.id;
        }
        if (this.hasNormals) {
            serializationObject.normals = Array.prototype.slice.call(this.getNormals());
        }
        if (this.hasTangents) {
            serializationObject.tangents = Array.prototype.slice.call(this.getTangents());
        }
        if (this.hasUVs) {
            serializationObject.uvs = Array.prototype.slice.call(this.getUVs());
        }
        if (this.hasUV2s) {
            serializationObject.uv2s = Array.prototype.slice.call(this.getUV2s());
        }
        if (this.hasColors) {
            serializationObject.colors = Array.prototype.slice.call(this.getColors());
        }
        // Animations
        SerializationHelper.AppendSerializedAnimations(this, serializationObject);
        return serializationObject;
    }
    /**
     * Returns the string "MorphTarget"
     * @returns "MorphTarget"
     */
    getClassName() {
        return "MorphTarget";
    }
    // Statics
    /**
     * Creates a new target from serialized data
     * @param serializationObject defines the serialized data to use
     * @param scene defines the hosting scene
     * @returns a new MorphTarget
     */
    static Parse(serializationObject, scene) {
        const result = new MorphTarget(serializationObject.name, serializationObject.influence);
        result.setPositions(serializationObject.positions);
        if (serializationObject.id != null) {
            result.id = serializationObject.id;
        }
        if (serializationObject.normals) {
            result.setNormals(serializationObject.normals);
        }
        if (serializationObject.tangents) {
            result.setTangents(serializationObject.tangents);
        }
        if (serializationObject.uvs) {
            result.setUVs(serializationObject.uvs);
        }
        if (serializationObject.uv2s) {
            result.setUV2s(serializationObject.uv2s);
        }
        if (serializationObject.colors) {
            result.setColors(serializationObject.colors);
        }
        // Animations
        if (serializationObject.animations) {
            for (let animationIndex = 0; animationIndex < serializationObject.animations.length; animationIndex++) {
                const parsedAnimation = serializationObject.animations[animationIndex];
                const internalClass = GetClass("BABYLON.Animation");
                if (internalClass) {
                    result.animations.push(internalClass.Parse(parsedAnimation));
                }
            }
            if (serializationObject.autoAnimate && scene) {
                scene.beginAnimation(result, serializationObject.autoAnimateFrom, serializationObject.autoAnimateTo, serializationObject.autoAnimateLoop, serializationObject.autoAnimateSpeed || 1.0);
            }
        }
        return result;
    }
    /**
     * Creates a MorphTarget from mesh data
     * @param mesh defines the source mesh
     * @param name defines the name to use for the new target
     * @param influence defines the influence to attach to the target
     * @returns a new MorphTarget
     */
    static FromMesh(mesh, name, influence) {
        if (!name) {
            name = mesh.name;
        }
        const result = new MorphTarget(name, influence, mesh.getScene());
        result.setPositions(mesh.getVerticesData(VertexBuffer.PositionKind));
        if (mesh.isVerticesDataPresent(VertexBuffer.NormalKind)) {
            result.setNormals(mesh.getVerticesData(VertexBuffer.NormalKind));
        }
        if (mesh.isVerticesDataPresent(VertexBuffer.TangentKind)) {
            result.setTangents(mesh.getVerticesData(VertexBuffer.TangentKind));
        }
        if (mesh.isVerticesDataPresent(VertexBuffer.UVKind)) {
            result.setUVs(mesh.getVerticesData(VertexBuffer.UVKind));
        }
        if (mesh.isVerticesDataPresent(VertexBuffer.UV2Kind)) {
            result.setUV2s(mesh.getVerticesData(VertexBuffer.UV2Kind));
        }
        if (mesh.isVerticesDataPresent(VertexBuffer.ColorKind)) {
            result.setColors(mesh.getVerticesData(VertexBuffer.ColorKind));
        }
        return result;
    }
}
__decorate([
    serialize()
], MorphTarget.prototype, "id", void 0);

/**
 * Class used to store 2D array textures containing user data
 */
class RawTexture2DArray extends Texture {
    /**
     * Gets the number of layers of the texture
     */
    get depth() {
        return this._depth;
    }
    /**
     * Create a new RawTexture2DArray
     * @param data defines the data of the texture
     * @param width defines the width of the texture
     * @param height defines the height of the texture
     * @param depth defines the number of layers of the texture
     * @param format defines the texture format to use
     * @param scene defines the hosting scene
     * @param generateMipMaps defines a boolean indicating if mip levels should be generated (true by default)
     * @param invertY defines if texture must be stored with Y axis inverted
     * @param samplingMode defines the sampling mode to use (Texture.TRILINEAR_SAMPLINGMODE by default)
     * @param textureType defines the texture Type (Engine.TEXTURETYPE_UNSIGNED_BYTE, Engine.TEXTURETYPE_FLOAT...)
     * @param creationFlags specific flags to use when creating the texture (1 for storage textures, for eg)
     */
    constructor(data, width, height, depth, 
    /** Gets or sets the texture format to use */
    format, scene, generateMipMaps = true, invertY = false, samplingMode = Texture.TRILINEAR_SAMPLINGMODE, textureType = 0, creationFlags) {
        super(null, scene, !generateMipMaps, invertY);
        this.format = format;
        this._texture = scene.getEngine().createRawTexture2DArray(data, width, height, depth, format, generateMipMaps, invertY, samplingMode, null, textureType, creationFlags);
        this._depth = depth;
        this.is2DArray = true;
    }
    /**
     * Update the texture with new data
     * @param data defines the data to store in the texture
     */
    update(data) {
        if (!this._texture) {
            return;
        }
        this._getEngine().updateRawTexture2DArray(this._texture, data, this._texture.format, this._texture.invertY, null, this._texture.type);
    }
    /**
     * Creates a RGBA texture from some data.
     * @param data Define the texture data
     * @param width Define the width of the texture
     * @param height Define the height of the texture
     * @param depth defines the number of layers of the texture
     * @param scene defines the scene the texture will belong to
     * @param generateMipMaps Define whether or not to create mip maps for the texture
     * @param invertY define if the data should be flipped on Y when uploaded to the GPU
     * @param samplingMode define the texture sampling mode (Texture.xxx_SAMPLINGMODE)
     * @param type define the format of the data (int, float... Engine.TEXTURETYPE_xxx)
     * @returns the RGBA texture
     */
    static CreateRGBATexture(data, width, height, depth, scene, generateMipMaps = true, invertY = false, samplingMode = 3, type = 0) {
        return new RawTexture2DArray(data, width, height, depth, 5, scene, generateMipMaps, invertY, samplingMode, type);
    }
}

/**
 * This class is used to deform meshes using morphing between different targets
 * @see https://doc.babylonjs.com/features/featuresDeepDive/mesh/morphTargets
 */
class MorphTargetManager {
    /**
     * Sets a boolean indicating that adding new target or updating an existing target will not update the underlying data buffers
     */
    set areUpdatesFrozen(block) {
        if (block) {
            this._blockCounter++;
        }
        else {
            this._blockCounter--;
            if (this._blockCounter <= 0) {
                this._blockCounter = 0;
                this._syncActiveTargets(this._forceUpdateWhenUnfrozen);
                this._forceUpdateWhenUnfrozen = false;
            }
        }
    }
    get areUpdatesFrozen() {
        return this._blockCounter > 0;
    }
    /**
     * Creates a new MorphTargetManager
     * @param scene defines the current scene
     */
    constructor(scene = null) {
        this._targets = new Array();
        this._targetInfluenceChangedObservers = new Array();
        this._targetDataLayoutChangedObservers = new Array();
        this._activeTargets = new SmartArray(16);
        this._supportsPositions = false;
        this._supportsNormals = false;
        this._supportsTangents = false;
        this._supportsUVs = false;
        this._supportsUV2s = false;
        this._supportsColors = false;
        this._vertexCount = 0;
        this._uniqueId = 0;
        this._tempInfluences = new Array();
        this._canUseTextureForTargets = false;
        this._blockCounter = 0;
        this._mustSynchronize = true;
        this._forceUpdateWhenUnfrozen = false;
        /** @internal */
        this._textureVertexStride = 0;
        /** @internal */
        this._textureWidth = 0;
        /** @internal */
        this._textureHeight = 1;
        /** @internal */
        this._parentContainer = null;
        /**
         * Gets or sets a boolean indicating if influencers must be optimized (eg. recompiling the shader if less influencers are used)
         */
        this.optimizeInfluencers = true;
        /**
         * Gets or sets a boolean indicating if positions must be morphed
         */
        this.enablePositionMorphing = true;
        /**
         * Gets or sets a boolean indicating if normals must be morphed
         */
        this.enableNormalMorphing = true;
        /**
         * Gets or sets a boolean indicating if tangents must be morphed
         */
        this.enableTangentMorphing = true;
        /**
         * Gets or sets a boolean indicating if UV must be morphed
         */
        this.enableUVMorphing = true;
        /**
         * Gets or sets a boolean indicating if UV2 must be morphed
         */
        this.enableUV2Morphing = true;
        /**
         * Gets or sets a boolean indicating if colors must be morphed
         */
        this.enableColorMorphing = true;
        this._numMaxInfluencers = 0;
        this._useTextureToStoreTargets = true;
        /**
         * Gets or sets an object used to store user defined information for the MorphTargetManager
         */
        this.metadata = null;
        this._influencesAreDirty = false;
        this._needUpdateInfluences = false;
        if (!scene) {
            scene = EngineStore.LastCreatedScene;
        }
        this._scene = scene;
        if (this._scene) {
            this._scene.addMorphTargetManager(this);
            this._uniqueId = this._scene.getUniqueId();
            const engineCaps = this._scene.getEngine().getCaps();
            this._canUseTextureForTargets =
                engineCaps.canUseGLVertexID && engineCaps.textureFloat && engineCaps.maxVertexTextureImageUnits > 0 && engineCaps.texture2DArrayMaxLayerCount > 1;
        }
    }
    /**
     * Gets or sets the maximum number of influencers (targets) (default value: 0).
     * Setting a value for this property can lead to a smoother experience, as only one shader will be compiled, which will use this value as the maximum number of influencers.
     * If you leave the value at 0 (default), a new shader will be compiled every time the number of active influencers changes. This can cause problems, as compiling a shader takes time.
     * If you assign a non-zero value to this property, you need to ensure that this value is greater than the maximum number of (active) influencers you'll need for this morph manager.
     * Otherwise, the number of active influencers will be truncated at the value you set for this property, which can lead to unexpected results.
     * Note that this property has no effect if "useTextureToStoreTargets" is false.
     * Note as well that if MorphTargetManager.ConstantTargetCountForTextureMode is greater than 0, this property will be ignored and the constant value will be used instead.
     */
    get numMaxInfluencers() {
        if (MorphTargetManager.ConstantTargetCountForTextureMode > 0 && this.isUsingTextureForTargets) {
            return MorphTargetManager.ConstantTargetCountForTextureMode;
        }
        return this._numMaxInfluencers;
    }
    set numMaxInfluencers(value) {
        if (this._numMaxInfluencers === value) {
            return;
        }
        this._numMaxInfluencers = value;
        this._mustSynchronize = true;
        this._syncActiveTargets();
    }
    /**
     * Gets the unique ID of this manager
     */
    get uniqueId() {
        return this._uniqueId;
    }
    /**
     * Gets the number of vertices handled by this manager
     */
    get vertexCount() {
        return this._vertexCount;
    }
    /**
     * Gets a boolean indicating if this manager supports morphing of positions
     */
    get supportsPositions() {
        return this._supportsPositions && this.enablePositionMorphing;
    }
    /**
     * Gets a boolean indicating if this manager supports morphing of normals
     */
    get supportsNormals() {
        return this._supportsNormals && this.enableNormalMorphing;
    }
    /**
     * Gets a boolean indicating if this manager supports morphing of tangents
     */
    get supportsTangents() {
        return this._supportsTangents && this.enableTangentMorphing;
    }
    /**
     * Gets a boolean indicating if this manager supports morphing of texture coordinates
     */
    get supportsUVs() {
        return this._supportsUVs && this.enableUVMorphing;
    }
    /**
     * Gets a boolean indicating if this manager supports morphing of texture coordinates 2
     */
    get supportsUV2s() {
        return this._supportsUV2s && this.enableUV2Morphing;
    }
    /**
     * Gets a boolean indicating if this manager supports morphing of colors
     */
    get supportsColors() {
        return this._supportsColors && this.enableColorMorphing;
    }
    /**
     * Gets a boolean indicating if this manager has data for morphing positions
     */
    get hasPositions() {
        return this._supportsPositions;
    }
    /**
     * Gets a boolean indicating if this manager has data for morphing normals
     */
    get hasNormals() {
        return this._supportsNormals;
    }
    /**
     * Gets a boolean indicating if this manager has data for morphing tangents
     */
    get hasTangents() {
        return this._supportsTangents;
    }
    /**
     * Gets a boolean indicating if this manager has data for morphing texture coordinates
     */
    get hasUVs() {
        return this._supportsUVs;
    }
    /**
     * Gets a boolean indicating if this manager has data for morphing texture coordinates 2
     */
    get hasUV2s() {
        return this._supportsUV2s;
    }
    /**
     * Gets a boolean indicating if this manager has data for morphing colors
     */
    get hasColors() {
        return this._supportsColors;
    }
    /**
     * Gets the number of targets stored in this manager
     */
    get numTargets() {
        return this._targets.length;
    }
    /**
     * Gets the number of influencers (ie. the number of targets with influences > 0)
     */
    get numInfluencers() {
        if (this._influencesAreDirty) {
            this._syncActiveTargets();
        }
        return this._activeTargets.length;
    }
    /**
     * Gets the list of influences (one per target)
     */
    get influences() {
        if (this._influencesAreDirty) {
            this._syncActiveTargets();
        }
        return this._influences;
    }
    /**
     * Gets or sets a boolean indicating that targets should be stored as a texture instead of using vertex attributes (default is true).
     * Please note that this option is not available if the hardware does not support it
     */
    get useTextureToStoreTargets() {
        return this._useTextureToStoreTargets;
    }
    set useTextureToStoreTargets(value) {
        if (this._useTextureToStoreTargets === value) {
            return;
        }
        this._useTextureToStoreTargets = value;
        this._mustSynchronize = true;
        this._syncActiveTargets();
    }
    /**
     * Gets a boolean indicating that the targets are stored into a texture (instead of as attributes)
     */
    get isUsingTextureForTargets() {
        return (MorphTargetManager.EnableTextureStorage &&
            this.useTextureToStoreTargets &&
            this._canUseTextureForTargets &&
            !this._scene?.getEngine().getCaps().disableMorphTargetTexture);
    }
    /**
     * Gets the active target at specified index. An active target is a target with an influence > 0
     * @param index defines the index to check
     * @returns the requested target
     */
    getActiveTarget(index) {
        if (this._influencesAreDirty) {
            this._syncActiveTargets();
        }
        return this._activeTargets.data[index];
    }
    /**
     * Gets the target at specified index
     * @param index defines the index to check
     * @returns the requested target
     */
    getTarget(index) {
        return this._targets[index];
    }
    /**
     * Gets the first target with the specified name
     * @param name defines the name to check
     * @returns the requested target
     */
    getTargetByName(name) {
        for (const target of this._targets) {
            if (target.name === name) {
                return target;
            }
        }
        return null;
    }
    /**
     * Add a new target to this manager
     * @param target defines the target to add
     */
    addTarget(target) {
        this._targets.push(target);
        this._targetInfluenceChangedObservers.push(target.onInfluenceChanged.add((needUpdate) => {
            if (this.areUpdatesFrozen && needUpdate) {
                this._forceUpdateWhenUnfrozen = true;
            }
            this._influencesAreDirty = true;
            this._needUpdateInfluences = this._needUpdateInfluences || needUpdate;
        }));
        this._targetDataLayoutChangedObservers.push(target._onDataLayoutChanged.add(() => {
            this._mustSynchronize = true;
            this._syncActiveTargets();
        }));
        this._mustSynchronize = true;
        this._syncActiveTargets();
    }
    /**
     * Removes a target from the manager
     * @param target defines the target to remove
     */
    removeTarget(target) {
        const index = this._targets.indexOf(target);
        if (index >= 0) {
            this._targets.splice(index, 1);
            target.onInfluenceChanged.remove(this._targetInfluenceChangedObservers.splice(index, 1)[0]);
            target._onDataLayoutChanged.remove(this._targetDataLayoutChangedObservers.splice(index, 1)[0]);
            this._mustSynchronize = true;
            this._syncActiveTargets();
        }
        if (this._scene) {
            this._scene.stopAnimation(target);
        }
    }
    /**
     * @internal
     */
    _bind(effect) {
        if (this._influencesAreDirty) {
            this._syncActiveTargets();
        }
        effect.setFloat3("morphTargetTextureInfo", this._textureVertexStride, this._textureWidth, this._textureHeight);
        effect.setFloatArray("morphTargetTextureIndices", this._morphTargetTextureIndices);
        effect.setTexture("morphTargets", this._targetStoreTexture);
        effect.setFloat("morphTargetCount", this.numInfluencers);
    }
    /**
     * Clone the current manager
     * @returns a new MorphTargetManager
     */
    clone() {
        const copy = new MorphTargetManager(this._scene);
        copy.areUpdatesFrozen = true;
        for (const target of this._targets) {
            copy.addTarget(target.clone());
        }
        copy.areUpdatesFrozen = false;
        copy.enablePositionMorphing = this.enablePositionMorphing;
        copy.enableNormalMorphing = this.enableNormalMorphing;
        copy.enableTangentMorphing = this.enableTangentMorphing;
        copy.enableUVMorphing = this.enableUVMorphing;
        copy.enableUV2Morphing = this.enableUV2Morphing;
        copy.enableColorMorphing = this.enableColorMorphing;
        copy.metadata = this.metadata;
        return copy;
    }
    /**
     * Serializes the current manager into a Serialization object
     * @returns the serialized object
     */
    serialize() {
        const serializationObject = {};
        serializationObject.id = this.uniqueId;
        serializationObject.targets = [];
        for (const target of this._targets) {
            serializationObject.targets.push(target.serialize());
        }
        if (this.metadata) {
            serializationObject.metadata = this.metadata;
        }
        return serializationObject;
    }
    _syncActiveTargets(needUpdate = false) {
        if (this.areUpdatesFrozen) {
            return;
        }
        needUpdate = needUpdate || this._needUpdateInfluences;
        this._needUpdateInfluences = false;
        this._influencesAreDirty = false;
        const wasUsingTextureForTargets = !!this._targetStoreTexture;
        const isUsingTextureForTargets = this.isUsingTextureForTargets;
        if (this._mustSynchronize || wasUsingTextureForTargets !== isUsingTextureForTargets) {
            this._mustSynchronize = false;
            this.synchronize();
        }
        let influenceCount = 0;
        this._activeTargets.reset();
        if (!this._morphTargetTextureIndices || this._morphTargetTextureIndices.length !== this._targets.length) {
            this._morphTargetTextureIndices = new Float32Array(this._targets.length);
        }
        let targetIndex = -1;
        for (const target of this._targets) {
            targetIndex++;
            if (target.influence === 0 && this.optimizeInfluencers) {
                continue;
            }
            if (this._activeTargets.length >= MorphTargetManager.MaxActiveMorphTargetsInVertexAttributeMode && !this.isUsingTextureForTargets) {
                break;
            }
            this._activeTargets.push(target);
            this._morphTargetTextureIndices[influenceCount] = targetIndex;
            this._tempInfluences[influenceCount++] = target.influence;
        }
        if (this._morphTargetTextureIndices.length !== influenceCount) {
            this._morphTargetTextureIndices = this._morphTargetTextureIndices.slice(0, influenceCount);
        }
        if (!this._influences || this._influences.length !== influenceCount) {
            this._influences = new Float32Array(influenceCount);
        }
        for (let index = 0; index < influenceCount; index++) {
            this._influences[index] = this._tempInfluences[index];
        }
        if (needUpdate && this._scene) {
            for (const mesh of this._scene.meshes) {
                if (mesh.morphTargetManager === this) {
                    if (isUsingTextureForTargets) {
                        mesh._markSubMeshesAsAttributesDirty();
                    }
                    else {
                        mesh._syncGeometryWithMorphTargetManager();
                    }
                }
            }
        }
    }
    /**
     * Synchronize the targets with all the meshes using this morph target manager
     */
    synchronize() {
        if (!this._scene || this.areUpdatesFrozen) {
            return;
        }
        const engine = this._scene.getEngine();
        this._supportsPositions = true;
        this._supportsNormals = true;
        this._supportsTangents = true;
        this._supportsUVs = true;
        this._supportsUV2s = true;
        this._supportsColors = true;
        this._vertexCount = 0;
        this._targetStoreTexture?.dispose();
        this._targetStoreTexture = null;
        if (this.isUsingTextureForTargets && this._targets.length > engine.getCaps().texture2DArrayMaxLayerCount) {
            this.useTextureToStoreTargets = false;
        }
        for (const target of this._targets) {
            this._supportsPositions = this._supportsPositions && target.hasPositions;
            this._supportsNormals = this._supportsNormals && target.hasNormals;
            this._supportsTangents = this._supportsTangents && target.hasTangents;
            this._supportsUVs = this._supportsUVs && target.hasUVs;
            this._supportsUV2s = this._supportsUV2s && target.hasUV2s;
            this._supportsColors = this._supportsColors && target.hasColors;
            const vertexCount = target.vertexCount;
            if (this._vertexCount === 0) {
                this._vertexCount = vertexCount;
            }
            else if (this._vertexCount !== vertexCount) {
                Logger.Error(`Incompatible target. Targets must all have the same vertices count. Current vertex count: ${this._vertexCount}, vertex count for target "${target.name}": ${vertexCount}`);
                return;
            }
        }
        if (this.isUsingTextureForTargets) {
            this._textureVertexStride = 0;
            this._supportsPositions && this._textureVertexStride++;
            this._supportsNormals && this._textureVertexStride++;
            this._supportsTangents && this._textureVertexStride++;
            this._supportsUVs && this._textureVertexStride++;
            this._supportsUV2s && this._textureVertexStride++;
            this._supportsColors && this._textureVertexStride++;
            this._textureWidth = this._vertexCount * this._textureVertexStride || 1;
            this._textureHeight = 1;
            const maxTextureSize = engine.getCaps().maxTextureSize;
            if (this._textureWidth > maxTextureSize) {
                this._textureHeight = Math.ceil(this._textureWidth / maxTextureSize);
                this._textureWidth = maxTextureSize;
            }
            const targetCount = this._targets.length;
            const data = new Float32Array(targetCount * this._textureWidth * this._textureHeight * 4);
            let offset = 0;
            for (let index = 0; index < targetCount; index++) {
                const target = this._targets[index];
                const positions = target.getPositions();
                const normals = target.getNormals();
                const uvs = target.getUVs();
                const tangents = target.getTangents();
                const uv2s = target.getUV2s();
                const colors = target.getColors();
                offset = index * this._textureWidth * this._textureHeight * 4;
                for (let vertex = 0; vertex < this._vertexCount; vertex++) {
                    if (this._supportsPositions && positions) {
                        data[offset] = positions[vertex * 3];
                        data[offset + 1] = positions[vertex * 3 + 1];
                        data[offset + 2] = positions[vertex * 3 + 2];
                        offset += 4;
                    }
                    if (this._supportsNormals && normals) {
                        data[offset] = normals[vertex * 3];
                        data[offset + 1] = normals[vertex * 3 + 1];
                        data[offset + 2] = normals[vertex * 3 + 2];
                        offset += 4;
                    }
                    if (this._supportsUVs && uvs) {
                        data[offset] = uvs[vertex * 2];
                        data[offset + 1] = uvs[vertex * 2 + 1];
                        offset += 4;
                    }
                    if (this._supportsTangents && tangents) {
                        data[offset] = tangents[vertex * 3];
                        data[offset + 1] = tangents[vertex * 3 + 1];
                        data[offset + 2] = tangents[vertex * 3 + 2];
                        offset += 4;
                    }
                    if (this._supportsUV2s && uv2s) {
                        data[offset] = uv2s[vertex * 2];
                        data[offset + 1] = uv2s[vertex * 2 + 1];
                        offset += 4;
                    }
                    if (this._supportsColors && colors) {
                        data[offset] = colors[vertex * 4];
                        data[offset + 1] = colors[vertex * 4 + 1];
                        data[offset + 2] = colors[vertex * 4 + 2];
                        data[offset + 3] = colors[vertex * 4 + 3];
                        offset += 4;
                    }
                }
            }
            this._targetStoreTexture = RawTexture2DArray.CreateRGBATexture(data, this._textureWidth, this._textureHeight, targetCount, this._scene, false, false, 1, 1);
            this._targetStoreTexture.name = `Morph texture_${this.uniqueId}`;
        }
        // Flag meshes as dirty to resync with the active targets
        for (const mesh of this._scene.meshes) {
            if (mesh.morphTargetManager === this) {
                mesh._syncGeometryWithMorphTargetManager();
            }
        }
    }
    /**
     * Release all resources
     */
    dispose() {
        if (this._targetStoreTexture) {
            this._targetStoreTexture.dispose();
        }
        this._targetStoreTexture = null;
        this.metadata = null;
        // Remove from scene
        if (this._scene) {
            this._scene.removeMorphTargetManager(this);
            if (this._parentContainer) {
                const index = this._parentContainer.morphTargetManagers.indexOf(this);
                if (index > -1) {
                    this._parentContainer.morphTargetManagers.splice(index, 1);
                }
                this._parentContainer = null;
            }
            for (const morph of this._targets) {
                this._scene.stopAnimation(morph);
            }
        }
    }
    // Statics
    /**
     * Creates a new MorphTargetManager from serialized data
     * @param serializationObject defines the serialized data
     * @param scene defines the hosting scene
     * @returns the new MorphTargetManager
     */
    static Parse(serializationObject, scene) {
        const result = new MorphTargetManager(scene);
        for (const targetData of serializationObject.targets) {
            result.addTarget(MorphTarget.Parse(targetData, scene));
        }
        if (serializationObject.metadata) {
            result.metadata = serializationObject.metadata;
        }
        return result;
    }
}
/** Enable storing morph target data into textures when set to true (true by default) */
MorphTargetManager.EnableTextureStorage = true;
/** Maximum number of active morph targets supported in the "vertex attribute" mode (i.e., not the "texture" mode) */
MorphTargetManager.MaxActiveMorphTargetsInVertexAttributeMode = 8;
/**
 * When used in texture mode, if greather than 0, this will override the the morph manager numMaxInfluencers value.
 */
MorphTargetManager.ConstantTargetCountForTextureMode = 0;

/**
 * Utility class for reading from a data buffer
 */
class DataReader {
    /**
     * Constructor
     * @param buffer The buffer to read
     */
    constructor(buffer) {
        /**
         * The current byte offset from the beginning of the data buffer.
         */
        this.byteOffset = 0;
        this.buffer = buffer;
    }
    /**
     * Loads the given byte length.
     * @param byteLength The byte length to load
     * @returns A promise that resolves when the load is complete
     */
    async loadAsync(byteLength) {
        const data = await this.buffer.readAsync(this.byteOffset, byteLength);
        this._dataView = new DataView(data.buffer, data.byteOffset, data.byteLength);
        this._dataByteOffset = 0;
    }
    /**
     * Read a unsigned 32-bit integer from the currently loaded data range.
     * @returns The 32-bit integer read
     */
    readUint32() {
        const value = this._dataView.getUint32(this._dataByteOffset, true);
        this._dataByteOffset += 4;
        this.byteOffset += 4;
        return value;
    }
    /**
     * Read a byte array from the currently loaded data range.
     * @param byteLength The byte length to read
     * @returns The byte array read
     */
    readUint8Array(byteLength) {
        const value = new Uint8Array(this._dataView.buffer, this._dataView.byteOffset + this._dataByteOffset, byteLength);
        this._dataByteOffset += byteLength;
        this.byteOffset += byteLength;
        return value;
    }
    /**
     * Read a string from the currently loaded data range.
     * @param byteLength The byte length to read
     * @returns The string read
     */
    readString(byteLength) {
        return Decode(this.readUint8Array(byteLength));
    }
    /**
     * Skips the given byte length the currently loaded data range.
     * @param byteLength The byte length to skip
     */
    skipBytes(byteLength) {
        this._dataByteOffset += byteLength;
        this.byteOffset += byteLength;
    }
}

function ValidateAsync(data, rootUrl, fileName, getExternalResource) {
    const options = {
        externalResourceFunction: getExternalResource,
    };
    if (fileName) {
        options.uri = rootUrl === "file:" ? fileName : rootUrl + fileName;
    }
    return ArrayBuffer.isView(data) ? GLTFValidator.validateBytes(data, options) : GLTFValidator.validateString(data, options);
}
/**
 * The worker function that gets converted to a blob url to pass into a worker.
 */
function WorkerFunc() {
    const pendingExternalResources = [];
    onmessage = (message) => {
        const data = message.data;
        switch (data.id) {
            case "init": {
                importScripts(data.url);
                break;
            }
            case "validate": {
                ValidateAsync(data.data, data.rootUrl, data.fileName, (uri) => new Promise((resolve, reject) => {
                    const index = pendingExternalResources.length;
                    pendingExternalResources.push({ resolve, reject });
                    postMessage({ id: "getExternalResource", index: index, uri: uri });
                })).then((value) => {
                    postMessage({ id: "validate.resolve", value: value });
                }, (reason) => {
                    postMessage({ id: "validate.reject", reason: reason });
                });
                break;
            }
            case "getExternalResource.resolve": {
                pendingExternalResources[data.index].resolve(data.value);
                break;
            }
            case "getExternalResource.reject": {
                pendingExternalResources[data.index].reject(data.reason);
                break;
            }
        }
    };
}
/**
 * glTF validation
 */
class GLTFValidation {
    /**
     * Validate a glTF asset using the glTF-Validator.
     * @param data The JSON of a glTF or the array buffer of a binary glTF
     * @param rootUrl The root url for the glTF
     * @param fileName The file name for the glTF
     * @param getExternalResource The callback to get external resources for the glTF validator
     * @returns A promise that resolves with the glTF validation results once complete
     */
    static ValidateAsync(data, rootUrl, fileName, getExternalResource) {
        if (typeof Worker === "function") {
            return new Promise((resolve, reject) => {
                const workerContent = `${ValidateAsync}(${WorkerFunc})()`;
                const workerBlobUrl = URL.createObjectURL(new Blob([workerContent], { type: "application/javascript" }));
                const worker = new Worker(workerBlobUrl);
                const onError = (error) => {
                    worker.removeEventListener("error", onError);
                    worker.removeEventListener("message", onMessage);
                    // eslint-disable-next-line @typescript-eslint/prefer-promise-reject-errors
                    reject(error);
                };
                const onMessage = (message) => {
                    const data = message.data;
                    switch (data.id) {
                        case "getExternalResource": {
                            getExternalResource(data.uri).then((value) => {
                                worker.postMessage({ id: "getExternalResource.resolve", index: data.index, value: value }, [value.buffer]);
                            }, (reason) => {
                                worker.postMessage({ id: "getExternalResource.reject", index: data.index, reason: reason });
                            });
                            break;
                        }
                        case "validate.resolve": {
                            worker.removeEventListener("error", onError);
                            worker.removeEventListener("message", onMessage);
                            resolve(data.value);
                            worker.terminate();
                            break;
                        }
                        case "validate.reject": {
                            worker.removeEventListener("error", onError);
                            worker.removeEventListener("message", onMessage);
                            // eslint-disable-next-line @typescript-eslint/prefer-promise-reject-errors
                            reject(data.reason);
                            worker.terminate();
                        }
                    }
                };
                worker.addEventListener("error", onError);
                worker.addEventListener("message", onMessage);
                worker.postMessage({ id: "init", url: Tools.GetBabylonScriptURL(this.Configuration.url) });
                if (ArrayBuffer.isView(data)) {
                    // Slice the data to avoid copying the whole array buffer.
                    const slicedData = data.slice();
                    worker.postMessage({ id: "validate", data: slicedData, rootUrl: rootUrl, fileName: fileName }, [slicedData.buffer]);
                }
                else {
                    worker.postMessage({ id: "validate", data: data, rootUrl: rootUrl, fileName: fileName });
                }
            });
        }
        else {
            if (!this._LoadScriptPromise) {
                this._LoadScriptPromise = Tools.LoadBabylonScriptAsync(this.Configuration.url);
            }
            return this._LoadScriptPromise.then(() => {
                return ValidateAsync(data, rootUrl, fileName, getExternalResource);
            });
        }
    }
}
/**
 * The configuration. Defaults to `{ url: "https://cdn.babylonjs.com/gltf_validator.js" }`.
 */
GLTFValidation.Configuration = {
    url: `${Tools._DefaultCdnUrl}/gltf_validator.js`,
};

function readAsync(arrayBuffer, byteOffset, byteLength) {
    try {
        return Promise.resolve(new Uint8Array(arrayBuffer, byteOffset, byteLength));
    }
    catch (e) {
        return Promise.reject(e);
    }
}
function readViewAsync(arrayBufferView, byteOffset, byteLength) {
    try {
        if (byteOffset < 0 || byteOffset >= arrayBufferView.byteLength) {
            throw new RangeError("Offset is out of range.");
        }
        if (byteOffset + byteLength > arrayBufferView.byteLength) {
            throw new RangeError("Length is out of range.");
        }
        return Promise.resolve(new Uint8Array(arrayBufferView.buffer, arrayBufferView.byteOffset + byteOffset, byteLength));
    }
    catch (e) {
        return Promise.reject(e);
    }
}
/**
 * Mode that determines the coordinate system to use.
 */
var GLTFLoaderCoordinateSystemMode;
(function (GLTFLoaderCoordinateSystemMode) {
    /**
     * Automatically convert the glTF right-handed data to the appropriate system based on the current coordinate system mode of the scene.
     */
    GLTFLoaderCoordinateSystemMode[GLTFLoaderCoordinateSystemMode["AUTO"] = 0] = "AUTO";
    /**
     * Sets the useRightHandedSystem flag on the scene.
     */
    GLTFLoaderCoordinateSystemMode[GLTFLoaderCoordinateSystemMode["FORCE_RIGHT_HANDED"] = 1] = "FORCE_RIGHT_HANDED";
})(GLTFLoaderCoordinateSystemMode || (GLTFLoaderCoordinateSystemMode = {}));
/**
 * Mode that determines what animations will start.
 */
var GLTFLoaderAnimationStartMode;
(function (GLTFLoaderAnimationStartMode) {
    /**
     * No animation will start.
     */
    GLTFLoaderAnimationStartMode[GLTFLoaderAnimationStartMode["NONE"] = 0] = "NONE";
    /**
     * The first animation will start.
     */
    GLTFLoaderAnimationStartMode[GLTFLoaderAnimationStartMode["FIRST"] = 1] = "FIRST";
    /**
     * All animations will start.
     */
    GLTFLoaderAnimationStartMode[GLTFLoaderAnimationStartMode["ALL"] = 2] = "ALL";
})(GLTFLoaderAnimationStartMode || (GLTFLoaderAnimationStartMode = {}));
/**
 * Loader state.
 */
var GLTFLoaderState;
(function (GLTFLoaderState) {
    /**
     * The asset is loading.
     */
    GLTFLoaderState[GLTFLoaderState["LOADING"] = 0] = "LOADING";
    /**
     * The asset is ready for rendering.
     */
    GLTFLoaderState[GLTFLoaderState["READY"] = 1] = "READY";
    /**
     * The asset is completely loaded.
     */
    GLTFLoaderState[GLTFLoaderState["COMPLETE"] = 2] = "COMPLETE";
})(GLTFLoaderState || (GLTFLoaderState = {}));
/**
 * This class contains all the concrete (not abstract) glTF options, excluding callbacks.
 * The purpose of this class is to make it easy to provide a way to mutate the default
 * loader options (see the GLTFLoaderDefaultOptions instance below) without duplicating
 * all the options in yet another object. Since this class is instantiated for the default
 * options object, abstract properties and callbacks are not included, it's more just
 * flag-type options.
 */
class GLTFLoaderBaseOptions {
    constructor() {
        /**
         * Defines if the loader should always compute the bounding boxes of meshes and not use the min/max values from the position accessor. Defaults to false.
         */
        this.alwaysComputeBoundingBox = false;
        /**
         * Defines if the loader should always compute the nearest common ancestor of the skeleton joints instead of using `skin.skeleton`. Defaults to false.
         * Set this to true if loading assets with invalid `skin.skeleton` values.
         */
        this.alwaysComputeSkeletonRootNode = false;
        /**
         * The animation start mode. Defaults to FIRST.
         */
        this.animationStartMode = GLTFLoaderAnimationStartMode.FIRST;
        /**
         * Defines if the loader should compile materials before raising the success callback. Defaults to false.
         */
        this.compileMaterials = false;
        /**
         * Defines if the loader should compile shadow generators before raising the success callback. Defaults to false.
         */
        this.compileShadowGenerators = false;
        /**
         * The coordinate system mode. Defaults to AUTO.
         */
        this.coordinateSystemMode = GLTFLoaderCoordinateSystemMode.AUTO;
        /**
         * Defines if the loader should create instances when multiple glTF nodes point to the same glTF mesh. Defaults to true.
         */
        this.createInstances = true;
        /**
         * If true, load all materials defined in the file, even if not used by any mesh. Defaults to false.
         */
        this.loadAllMaterials = false;
        /**
         * Defines if the loader should load morph targets. Defaults to true.
         */
        this.loadMorphTargets = true;
        /**
         * Defines if the loader should load node animations. Defaults to true.
         * NOTE: The animation of this node will still load if the node is also a joint of a skin and `loadSkins` is true.
         */
        this.loadNodeAnimations = true;
        /**
         * If true, load only the materials defined in the file. Defaults to false.
         */
        this.loadOnlyMaterials = false;
        /**
         * Defines if the loader should load skins. Defaults to true.
         */
        this.loadSkins = true;
        /**
         * If true, do not load any materials defined in the file. Defaults to false.
         */
        this.skipMaterials = false;
        /**
         * When loading glTF animations, which are defined in seconds, target them to this FPS. Defaults to 60.
         */
        this.targetFps = 60;
        /**
         * Defines if the Alpha blended materials are only applied as coverage.
         * If false, (default) The luminance of each pixel will reduce its opacity to simulate the behaviour of most physical materials.
         * If true, no extra effects are applied to transparent pixels.
         */
        this.transparencyAsCoverage = false;
        /**
         * Defines if the loader should also compile materials with clip planes. Defaults to false.
         */
        this.useClipPlane = false;
        /**
         * If true, the loader will derive the name for Babylon textures from the glTF texture name, image name, or image url. Defaults to false.
         * Note that it is possible for multiple Babylon textures to share the same name when the Babylon textures load from the same glTF texture or image.
         */
        this.useGltfTextureNames = false;
        /**
         * Defines if the loader should use range requests when load binary glTF files from HTTP.
         * Enabling will disable offline support and glTF validator.
         * Defaults to false.
         */
        this.useRangeRequests = false;
        /**
         * If true, load the color (gamma encoded) textures into sRGB buffers (if supported by the GPU), which will yield more accurate results when sampling the texture. Defaults to true.
         */
        this.useSRGBBuffers = true;
        /**
         * Defines if the loader should validate the asset.
         */
        this.validate = false;
        /**
         * Load the glTF files using the OpenPBR material.
         * @experimental
         */
        this.useOpenPBR = false;
        /**
         * If true, the loader will not use the transmission helper when loading materials with transmission.
         */
        this.dontUseTransmissionHelper = false;
    }
}
/**
 * The default GLTF loader options.
 * Override the properties of this object to globally change the default loader options.
 * To specify options for a specific load call, pass those options into the associated load function.
 */
const GLTFLoaderDefaultOptions = new GLTFLoaderBaseOptions();
class GLTFLoaderOptions extends GLTFLoaderBaseOptions {
    constructor() {
        super(...arguments);
        /**
         * Defines options for glTF extensions.
         */
        this.extensionOptions = {};
        /**
         * Function called before loading a url referenced by the asset.
         * @param url url referenced by the asset
         * @returns Async url to load
         */
        this.preprocessUrlAsync = (url) => Promise.resolve(url);
    }
    // eslint-disable-next-line babylonjs/available
    copyFrom(options) {
        if (options) {
            this.alwaysComputeBoundingBox = options.alwaysComputeBoundingBox ?? this.alwaysComputeBoundingBox;
            this.alwaysComputeSkeletonRootNode = options.alwaysComputeSkeletonRootNode ?? this.alwaysComputeSkeletonRootNode;
            this.animationStartMode = options.animationStartMode ?? this.animationStartMode;
            this.capturePerformanceCounters = options.capturePerformanceCounters ?? this.capturePerformanceCounters;
            this.compileMaterials = options.compileMaterials ?? this.compileMaterials;
            this.compileShadowGenerators = options.compileShadowGenerators ?? this.compileShadowGenerators;
            this.coordinateSystemMode = options.coordinateSystemMode ?? this.coordinateSystemMode;
            this.createInstances = options.createInstances ?? this.createInstances;
            this.customRootNode = options.customRootNode;
            this.extensionOptions = options.extensionOptions ?? this.extensionOptions;
            this.loadAllMaterials = options.loadAllMaterials ?? this.loadAllMaterials;
            this.loadMorphTargets = options.loadMorphTargets ?? this.loadMorphTargets;
            this.loadNodeAnimations = options.loadNodeAnimations ?? this.loadNodeAnimations;
            this.loadOnlyMaterials = options.loadOnlyMaterials ?? this.loadOnlyMaterials;
            this.loadSkins = options.loadSkins ?? this.loadSkins;
            this.loggingEnabled = options.loggingEnabled ?? this.loggingEnabled;
            this.onCameraLoaded = options.onCameraLoaded;
            this.onMaterialLoaded = options.onMaterialLoaded;
            this.onMeshLoaded = options.onMeshLoaded;
            this.onParsed = options.onParsed;
            this.onSkinLoaded = options.onSkinLoaded;
            this.onTextureLoaded = options.onTextureLoaded;
            this.onValidated = options.onValidated;
            this.preprocessUrlAsync = options.preprocessUrlAsync ?? this.preprocessUrlAsync;
            this.skipMaterials = options.skipMaterials ?? this.skipMaterials;
            this.targetFps = options.targetFps ?? this.targetFps;
            this.transparencyAsCoverage = options.transparencyAsCoverage ?? this.transparencyAsCoverage;
            this.useClipPlane = options.useClipPlane ?? this.useClipPlane;
            this.useGltfTextureNames = options.useGltfTextureNames ?? this.useGltfTextureNames;
            this.useOpenPBR = options.useOpenPBR ?? this.useOpenPBR;
            this.useRangeRequests = options.useRangeRequests ?? this.useRangeRequests;
            this.useSRGBBuffers = options.useSRGBBuffers ?? this.useSRGBBuffers;
            this.validate = options.validate ?? this.validate;
            this.dontUseTransmissionHelper = options.dontUseTransmissionHelper ?? this.dontUseTransmissionHelper;
        }
    }
}
/**
 * File loader for loading glTF files into a scene.
 */
class GLTFFileLoader extends GLTFLoaderOptions {
    /**
     * Creates a new glTF file loader.
     * @param options The options for the loader
     */
    constructor(options) {
        super();
        // --------------------
        // Begin Common options
        // --------------------
        /**
         * Raised when the asset has been parsed
         */
        this.onParsedObservable = new Observable();
        // --------------
        // End V1 options
        // --------------
        /**
         * Observable raised when the loader creates a mesh after parsing the glTF properties of the mesh.
         * Note that the observable is raised as soon as the mesh object is created, meaning some data may not have been setup yet for this mesh (vertex data, morph targets, material, ...)
         */
        this.onMeshLoadedObservable = new Observable();
        /**
         * Observable raised when the loader creates a skin after parsing the glTF properties of the skin node.
         * @see https://doc.babylonjs.com/features/featuresDeepDive/importers/glTF/glTFSkinning#ignoring-the-transform-of-the-skinned-mesh
         * @param node - the transform node that corresponds to the original glTF skin node used for animations
         * @param skinnedNode - the transform node that is the skinned mesh itself or the parent of the skinned meshes
         */
        this.onSkinLoadedObservable = new Observable();
        /**
         * Observable raised when the loader creates a texture after parsing the glTF properties of the texture.
         */
        this.onTextureLoadedObservable = new Observable();
        /**
         * Observable raised when the loader creates a material after parsing the glTF properties of the material.
         */
        this.onMaterialLoadedObservable = new Observable();
        /**
         * Observable raised when the loader creates a camera after parsing the glTF properties of the camera.
         */
        this.onCameraLoadedObservable = new Observable();
        /**
         * Observable raised when the asset is completely loaded, immediately before the loader is disposed.
         * For assets with LODs, raised when all of the LODs are complete.
         * For assets without LODs, raised when the model is complete, immediately after the loader resolves the returned promise.
         */
        this.onCompleteObservable = new Observable();
        /**
         * Observable raised when an error occurs.
         */
        this.onErrorObservable = new Observable();
        /**
         * Observable raised after the loader is disposed.
         */
        this.onDisposeObservable = new Observable();
        /**
         * Observable raised after a loader extension is created.
         * Set additional options for a loader extension in this event.
         */
        this.onExtensionLoadedObservable = new Observable();
        /**
         * Observable raised after validation when validate is set to true. The event data is the result of the validation.
         */
        this.onValidatedObservable = new Observable();
        this._loader = null;
        this._state = null;
        this._requests = new Array();
        /**
         * Name of the loader ("gltf")
         */
        this.name = GLTFFileLoaderMetadata.name;
        /** @internal */
        this.extensions = GLTFFileLoaderMetadata.extensions;
        /**
         * Observable raised when the loader state changes.
         */
        this.onLoaderStateChangedObservable = new Observable();
        this._logIndentLevel = 0;
        this._loggingEnabled = false;
        /** @internal */
        this._log = this._logDisabled;
        this._capturePerformanceCounters = false;
        /** @internal */
        this._startPerformanceCounter = this._startPerformanceCounterDisabled;
        /** @internal */
        this._endPerformanceCounter = this._endPerformanceCounterDisabled;
        this.copyFrom(Object.assign({ ...GLTFLoaderDefaultOptions }, options));
    }
    /**
     * Raised when the asset has been parsed
     */
    set onParsed(callback) {
        if (this._onParsedObserver) {
            this.onParsedObservable.remove(this._onParsedObserver);
        }
        if (callback) {
            this._onParsedObserver = this.onParsedObservable.add(callback);
        }
    }
    /**
     * Callback raised when the loader creates a mesh after parsing the glTF properties of the mesh.
     * Note that the callback is called as soon as the mesh object is created, meaning some data may not have been setup yet for this mesh (vertex data, morph targets, material, ...)
     */
    set onMeshLoaded(callback) {
        if (this._onMeshLoadedObserver) {
            this.onMeshLoadedObservable.remove(this._onMeshLoadedObserver);
        }
        if (callback) {
            this._onMeshLoadedObserver = this.onMeshLoadedObservable.add(callback);
        }
    }
    /**
     * Callback raised when the loader creates a skin after parsing the glTF properties of the skin node.
     * @see https://doc.babylonjs.com/features/featuresDeepDive/importers/glTF/glTFSkinning#ignoring-the-transform-of-the-skinned-mesh
     */
    set onSkinLoaded(callback) {
        if (this._onSkinLoadedObserver) {
            this.onSkinLoadedObservable.remove(this._onSkinLoadedObserver);
        }
        if (callback) {
            this._onSkinLoadedObserver = this.onSkinLoadedObservable.add((data) => callback(data.node, data.skinnedNode));
        }
    }
    /**
     * Callback raised when the loader creates a texture after parsing the glTF properties of the texture.
     */
    set onTextureLoaded(callback) {
        if (this._onTextureLoadedObserver) {
            this.onTextureLoadedObservable.remove(this._onTextureLoadedObserver);
        }
        if (callback) {
            this._onTextureLoadedObserver = this.onTextureLoadedObservable.add(callback);
        }
    }
    /**
     * Callback raised when the loader creates a material after parsing the glTF properties of the material.
     */
    set onMaterialLoaded(callback) {
        if (this._onMaterialLoadedObserver) {
            this.onMaterialLoadedObservable.remove(this._onMaterialLoadedObserver);
        }
        if (callback) {
            this._onMaterialLoadedObserver = this.onMaterialLoadedObservable.add(callback);
        }
    }
    /**
     * Callback raised when the loader creates a camera after parsing the glTF properties of the camera.
     */
    set onCameraLoaded(callback) {
        if (this._onCameraLoadedObserver) {
            this.onCameraLoadedObservable.remove(this._onCameraLoadedObserver);
        }
        if (callback) {
            this._onCameraLoadedObserver = this.onCameraLoadedObservable.add(callback);
        }
    }
    /**
     * Callback raised when the asset is completely loaded, immediately before the loader is disposed.
     * For assets with LODs, raised when all of the LODs are complete.
     * For assets without LODs, raised when the model is complete, immediately after the loader resolves the returned promise.
     */
    set onComplete(callback) {
        if (this._onCompleteObserver) {
            this.onCompleteObservable.remove(this._onCompleteObserver);
        }
        this._onCompleteObserver = this.onCompleteObservable.add(callback);
    }
    /**
     * Callback raised when an error occurs.
     */
    set onError(callback) {
        if (this._onErrorObserver) {
            this.onErrorObservable.remove(this._onErrorObserver);
        }
        this._onErrorObserver = this.onErrorObservable.add(callback);
    }
    /**
     * Callback raised after the loader is disposed.
     */
    set onDispose(callback) {
        if (this._onDisposeObserver) {
            this.onDisposeObservable.remove(this._onDisposeObserver);
        }
        this._onDisposeObserver = this.onDisposeObservable.add(callback);
    }
    /**
     * Callback raised after a loader extension is created.
     */
    set onExtensionLoaded(callback) {
        if (this._onExtensionLoadedObserver) {
            this.onExtensionLoadedObservable.remove(this._onExtensionLoadedObserver);
        }
        this._onExtensionLoadedObserver = this.onExtensionLoadedObservable.add(callback);
    }
    /**
     * Defines if the loader logging is enabled.
     */
    get loggingEnabled() {
        return this._loggingEnabled;
    }
    set loggingEnabled(value) {
        if (this._loggingEnabled === value) {
            return;
        }
        this._loggingEnabled = value;
        if (this._loggingEnabled) {
            this._log = this._logEnabled;
        }
        else {
            this._log = this._logDisabled;
        }
    }
    /**
     * Defines if the loader should capture performance counters.
     */
    get capturePerformanceCounters() {
        return this._capturePerformanceCounters;
    }
    set capturePerformanceCounters(value) {
        if (this._capturePerformanceCounters === value) {
            return;
        }
        this._capturePerformanceCounters = value;
        if (this._capturePerformanceCounters) {
            this._startPerformanceCounter = this._startPerformanceCounterEnabled;
            this._endPerformanceCounter = this._endPerformanceCounterEnabled;
        }
        else {
            this._startPerformanceCounter = this._startPerformanceCounterDisabled;
            this._endPerformanceCounter = this._endPerformanceCounterDisabled;
        }
    }
    /**
     * Callback raised after the asset is validated.
     */
    set onValidated(callback) {
        if (this._onValidatedObserver) {
            this.onValidatedObservable.remove(this._onValidatedObserver);
        }
        this._onValidatedObserver = this.onValidatedObservable.add(callback);
    }
    /**
     * Disposes the loader, releases resources during load, and cancels any outstanding requests.
     */
    dispose() {
        if (this._loader) {
            this._loader.dispose();
            this._loader = null;
        }
        for (const request of this._requests) {
            request.abort();
        }
        this._requests.length = 0;
        delete this._progressCallback;
        this.preprocessUrlAsync = (url) => Promise.resolve(url);
        this.onMeshLoadedObservable.clear();
        this.onSkinLoadedObservable.clear();
        this.onTextureLoadedObservable.clear();
        this.onMaterialLoadedObservable.clear();
        this.onCameraLoadedObservable.clear();
        this.onCompleteObservable.clear();
        this.onExtensionLoadedObservable.clear();
        this.onDisposeObservable.notifyObservers(undefined);
        this.onDisposeObservable.clear();
    }
    /**
     * @internal
     */
    loadFile(scene, fileOrUrl, rootUrl, onSuccess, onProgress, useArrayBuffer, onError, name) {
        if (ArrayBuffer.isView(fileOrUrl)) {
            this._loadBinary(scene, fileOrUrl, rootUrl, onSuccess, onError, name);
            return null;
        }
        this._progressCallback = onProgress;
        const fileName = fileOrUrl.name || Tools.GetFilename(fileOrUrl);
        if (useArrayBuffer) {
            if (this.useRangeRequests) {
                if (this.validate) {
                    Logger.Warn("glTF validation is not supported when range requests are enabled");
                }
                const fileRequest = {
                    abort: () => { },
                    onCompleteObservable: new Observable(),
                };
                const dataBuffer = {
                    readAsync: (byteOffset, byteLength) => {
                        return new Promise((resolve, reject) => {
                            this._loadFile(scene, fileOrUrl, (data) => {
                                resolve(new Uint8Array(data));
                            }, true, (error) => {
                                reject(error);
                            }, (webRequest) => {
                                webRequest.setRequestHeader("Range", `bytes=${byteOffset}-${byteOffset + byteLength - 1}`);
                            });
                        });
                    },
                    byteLength: 0,
                };
                this._unpackBinaryAsync(new DataReader(dataBuffer)).then((loaderData) => {
                    fileRequest.onCompleteObservable.notifyObservers(fileRequest);
                    onSuccess(loaderData);
                }, onError ? (error) => onError(undefined, error) : undefined);
                return fileRequest;
            }
            return this._loadFile(scene, fileOrUrl, (data) => {
                this._validate(scene, new Uint8Array(data, 0, data.byteLength), rootUrl, fileName);
                this._unpackBinaryAsync(new DataReader({
                    readAsync: (byteOffset, byteLength) => readAsync(data, byteOffset, byteLength),
                    byteLength: data.byteLength,
                })).then((loaderData) => {
                    onSuccess(loaderData);
                }, onError ? (error) => onError(undefined, error) : undefined);
            }, true, onError);
        }
        else {
            return this._loadFile(scene, fileOrUrl, (data) => {
                try {
                    this._validate(scene, data, rootUrl, fileName);
                    onSuccess({ json: this._parseJson(data) });
                }
                catch {
                    if (onError) {
                        onError();
                    }
                }
            }, false, onError);
        }
    }
    _loadBinary(scene, data, rootUrl, onSuccess, onError, fileName) {
        this._validate(scene, new Uint8Array(data.buffer, data.byteOffset, data.byteLength), rootUrl, fileName);
        this._unpackBinaryAsync(new DataReader({
            readAsync: (byteOffset, byteLength) => readViewAsync(data, byteOffset, byteLength),
            byteLength: data.byteLength,
        })).then((loaderData) => {
            onSuccess(loaderData);
        }, onError ? (error) => onError(undefined, error) : undefined);
    }
    /**
     * @internal
     */
    importMeshAsync(meshesNames, scene, data, rootUrl, onProgress, fileName) {
        return Promise.resolve().then(() => {
            this.onParsedObservable.notifyObservers(data);
            this.onParsedObservable.clear();
            this._log(`Loading ${fileName || ""}`);
            this._loader = this._getLoader(data);
            return this._loader.importMeshAsync(meshesNames, scene, null, data, rootUrl, onProgress, fileName);
        });
    }
    /**
     * @internal
     */
    loadAsync(scene, data, rootUrl, onProgress, fileName) {
        return Promise.resolve().then(() => {
            this.onParsedObservable.notifyObservers(data);
            this.onParsedObservable.clear();
            this._log(`Loading ${fileName || ""}`);
            this._loader = this._getLoader(data);
            return this._loader.loadAsync(scene, data, rootUrl, onProgress, fileName);
        });
    }
    /**
     * @internal
     */
    loadAssetContainerAsync(scene, data, rootUrl, onProgress, fileName) {
        return Promise.resolve().then(() => {
            this.onParsedObservable.notifyObservers(data);
            this.onParsedObservable.clear();
            this._log(`Loading ${fileName || ""}`);
            this._loader = this._getLoader(data);
            // Prepare the asset container.
            const container = new AssetContainer(scene);
            // Get materials/textures when loading to add to container
            const materials = [];
            this.onMaterialLoadedObservable.add((material) => {
                materials.push(material);
            });
            const textures = [];
            this.onTextureLoadedObservable.add((texture) => {
                textures.push(texture);
            });
            const cameras = [];
            this.onCameraLoadedObservable.add((camera) => {
                cameras.push(camera);
            });
            const morphTargetManagers = [];
            this.onMeshLoadedObservable.add((mesh) => {
                if (mesh.morphTargetManager) {
                    morphTargetManagers.push(mesh.morphTargetManager);
                }
            });
            return this._loader.importMeshAsync(null, scene, container, data, rootUrl, onProgress, fileName).then((result) => {
                Array.prototype.push.apply(container.geometries, result.geometries);
                Array.prototype.push.apply(container.meshes, result.meshes);
                Array.prototype.push.apply(container.particleSystems, result.particleSystems);
                Array.prototype.push.apply(container.skeletons, result.skeletons);
                Array.prototype.push.apply(container.animationGroups, result.animationGroups);
                Array.prototype.push.apply(container.materials, materials);
                Array.prototype.push.apply(container.textures, textures);
                Array.prototype.push.apply(container.lights, result.lights);
                Array.prototype.push.apply(container.transformNodes, result.transformNodes);
                Array.prototype.push.apply(container.cameras, cameras);
                Array.prototype.push.apply(container.morphTargetManagers, morphTargetManagers);
                return container;
            });
        });
    }
    /**
     * @internal
     */
    canDirectLoad(data) {
        return GLTFFileLoaderMetadata.canDirectLoad(data);
    }
    /**
     * @internal
     */
    directLoad(scene, data) {
        if (data.startsWith("base64," + GLTFMagicBase64Encoded) || // this is technically incorrect, but will continue to support for backcompat.
            data.startsWith(";base64," + GLTFMagicBase64Encoded) ||
            data.startsWith("application/octet-stream;base64," + GLTFMagicBase64Encoded) ||
            data.startsWith("model/gltf-binary;base64," + GLTFMagicBase64Encoded)) {
            const arrayBuffer = DecodeBase64UrlToBinary(data);
            this._validate(scene, new Uint8Array(arrayBuffer, 0, arrayBuffer.byteLength));
            return this._unpackBinaryAsync(new DataReader({
                readAsync: (byteOffset, byteLength) => readAsync(arrayBuffer, byteOffset, byteLength),
                byteLength: arrayBuffer.byteLength,
            }));
        }
        this._validate(scene, data);
        return Promise.resolve({ json: this._parseJson(data) });
    }
    /** @internal */
    createPlugin(options) {
        return new GLTFFileLoader(options[GLTFFileLoaderMetadata.name]);
    }
    /**
     * The loader state or null if the loader is not active.
     */
    get loaderState() {
        return this._state;
    }
    /**
     * Returns a promise that resolves when the asset is completely loaded.
     * @returns a promise that resolves when the asset is completely loaded.
     */
    whenCompleteAsync() {
        return new Promise((resolve, reject) => {
            this.onCompleteObservable.addOnce(() => {
                resolve();
            });
            this.onErrorObservable.addOnce((reason) => {
                reject(reason);
            });
        });
    }
    /**
     * @internal
     */
    _setState(state) {
        if (this._state === state) {
            return;
        }
        this._state = state;
        this.onLoaderStateChangedObservable.notifyObservers(this._state);
        this._log(GLTFLoaderState[this._state]);
    }
    /**
     * @internal
     */
    _loadFile(scene, fileOrUrl, onSuccess, useArrayBuffer, onError, onOpened) {
        const request = scene._loadFile(fileOrUrl, onSuccess, (event) => {
            this._onProgress(event, request);
        }, true, useArrayBuffer, onError, onOpened);
        request.onCompleteObservable.add(() => {
            // Force the length computable to be true since we can guarantee the data is loaded.
            request._lengthComputable = true;
            request._total = request._loaded;
        });
        this._requests.push(request);
        return request;
    }
    _onProgress(event, request) {
        if (!this._progressCallback) {
            return;
        }
        request._lengthComputable = event.lengthComputable;
        request._loaded = event.loaded;
        request._total = event.total;
        let lengthComputable = true;
        let loaded = 0;
        let total = 0;
        for (const request of this._requests) {
            if (request._lengthComputable === undefined || request._loaded === undefined || request._total === undefined) {
                return;
            }
            lengthComputable = lengthComputable && request._lengthComputable;
            loaded += request._loaded;
            total += request._total;
        }
        this._progressCallback({
            lengthComputable: lengthComputable,
            loaded: loaded,
            total: lengthComputable ? total : 0,
        });
    }
    _validate(scene, data, rootUrl = "", fileName = "") {
        if (!this.validate) {
            return;
        }
        this._startPerformanceCounter("Validate JSON");
        GLTFValidation.ValidateAsync(data, rootUrl, fileName, (uri) => {
            return this.preprocessUrlAsync(rootUrl + uri).then((url) => {
                return scene._loadFileAsync(url, undefined, true, true).then((data) => {
                    return new Uint8Array(data, 0, data.byteLength);
                });
            });
        }).then((result) => {
            this._endPerformanceCounter("Validate JSON");
            this.onValidatedObservable.notifyObservers(result);
            this.onValidatedObservable.clear();
        }, (reason) => {
            this._endPerformanceCounter("Validate JSON");
            Tools.Warn(`Failed to validate: ${reason.message}`);
            this.onValidatedObservable.clear();
        });
    }
    _getLoader(loaderData) {
        const asset = loaderData.json.asset || {};
        this._log(`Asset version: ${asset.version}`);
        asset.minVersion && this._log(`Asset minimum version: ${asset.minVersion}`);
        asset.generator && this._log(`Asset generator: ${asset.generator}`);
        const version = GLTFFileLoader._parseVersion(asset.version);
        if (!version) {
            throw new Error("Invalid version: " + asset.version);
        }
        if (asset.minVersion !== undefined) {
            const minVersion = GLTFFileLoader._parseVersion(asset.minVersion);
            if (!minVersion) {
                throw new Error("Invalid minimum version: " + asset.minVersion);
            }
            if (GLTFFileLoader._compareVersion(minVersion, { major: 2, minor: 0 }) > 0) {
                throw new Error("Incompatible minimum version: " + asset.minVersion);
            }
        }
        const createLoaders = {
            1: GLTFFileLoader._CreateGLTF1Loader,
            2: GLTFFileLoader._CreateGLTF2Loader,
        };
        const createLoader = createLoaders[version.major];
        if (!createLoader) {
            throw new Error("Unsupported version: " + asset.version);
        }
        return createLoader(this);
    }
    _parseJson(json) {
        this._startPerformanceCounter("Parse JSON");
        this._log(`JSON length: ${json.length}`);
        const parsed = JSON.parse(json);
        this._endPerformanceCounter("Parse JSON");
        return parsed;
    }
    _unpackBinaryAsync(dataReader) {
        this._startPerformanceCounter("Unpack Binary");
        // Read magic + version + length + json length + json format
        return dataReader.loadAsync(20).then(() => {
            const Binary = {
                Magic: 0x46546c67,
            };
            const magic = dataReader.readUint32();
            if (magic !== Binary.Magic) {
                throw new RuntimeError("Unexpected magic: " + magic, ErrorCodes.GLTFLoaderUnexpectedMagicError);
            }
            const version = dataReader.readUint32();
            if (this.loggingEnabled) {
                this._log(`Binary version: ${version}`);
            }
            const length = dataReader.readUint32();
            if (!this.useRangeRequests && length !== dataReader.buffer.byteLength) {
                Logger.Warn(`Length in header does not match actual data length: ${length} != ${dataReader.buffer.byteLength}`);
            }
            let unpacked;
            switch (version) {
                case 1: {
                    unpacked = this._unpackBinaryV1Async(dataReader, length);
                    break;
                }
                case 2: {
                    unpacked = this._unpackBinaryV2Async(dataReader, length);
                    break;
                }
                default: {
                    throw new Error("Unsupported version: " + version);
                }
            }
            this._endPerformanceCounter("Unpack Binary");
            return unpacked;
        });
    }
    _unpackBinaryV1Async(dataReader, length) {
        const ContentFormat = {
            JSON: 0,
        };
        const contentLength = dataReader.readUint32();
        const contentFormat = dataReader.readUint32();
        if (contentFormat !== ContentFormat.JSON) {
            throw new Error(`Unexpected content format: ${contentFormat}`);
        }
        const bodyLength = length - dataReader.byteOffset;
        const data = { json: this._parseJson(dataReader.readString(contentLength)), bin: null };
        if (bodyLength !== 0) {
            const startByteOffset = dataReader.byteOffset;
            data.bin = {
                readAsync: (byteOffset, byteLength) => dataReader.buffer.readAsync(startByteOffset + byteOffset, byteLength),
                byteLength: bodyLength,
            };
        }
        return Promise.resolve(data);
    }
    _unpackBinaryV2Async(dataReader, length) {
        const ChunkFormat = {
            JSON: 0x4e4f534a,
            BIN: 0x004e4942,
        };
        // Read the JSON chunk header.
        const chunkLength = dataReader.readUint32();
        const chunkFormat = dataReader.readUint32();
        if (chunkFormat !== ChunkFormat.JSON) {
            throw new Error("First chunk format is not JSON");
        }
        // Bail if there are no other chunks.
        if (dataReader.byteOffset + chunkLength === length) {
            return dataReader.loadAsync(chunkLength).then(() => {
                return { json: this._parseJson(dataReader.readString(chunkLength)), bin: null };
            });
        }
        // Read the JSON chunk and the length and type of the next chunk.
        return dataReader.loadAsync(chunkLength + 8).then(() => {
            const data = { json: this._parseJson(dataReader.readString(chunkLength)), bin: null };
            const readAsync = () => {
                const chunkLength = dataReader.readUint32();
                const chunkFormat = dataReader.readUint32();
                switch (chunkFormat) {
                    case ChunkFormat.JSON: {
                        throw new Error("Unexpected JSON chunk");
                    }
                    case ChunkFormat.BIN: {
                        const startByteOffset = dataReader.byteOffset;
                        data.bin = {
                            readAsync: (byteOffset, byteLength) => dataReader.buffer.readAsync(startByteOffset + byteOffset, byteLength),
                            byteLength: chunkLength,
                        };
                        dataReader.skipBytes(chunkLength);
                        break;
                    }
                    default: {
                        // ignore unrecognized chunkFormat
                        dataReader.skipBytes(chunkLength);
                        break;
                    }
                }
                if (dataReader.byteOffset !== length) {
                    return dataReader.loadAsync(8).then(readAsync);
                }
                return Promise.resolve(data);
            };
            return readAsync();
        });
    }
    static _parseVersion(version) {
        if (version === "1.0" || version === "1.0.1") {
            return {
                major: 1,
                minor: 0,
            };
        }
        const match = (version + "").match(/^(\d+)\.(\d+)/);
        if (!match) {
            return null;
        }
        return {
            major: parseInt(match[1]),
            minor: parseInt(match[2]),
        };
    }
    static _compareVersion(a, b) {
        if (a.major > b.major) {
            return 1;
        }
        if (a.major < b.major) {
            return -1;
        }
        if (a.minor > b.minor) {
            return 1;
        }
        if (a.minor < b.minor) {
            return -1;
        }
        return 0;
    }
    /**
     * @internal
     */
    _logOpen(message) {
        this._log(message);
        this._logIndentLevel++;
    }
    /** @internal */
    _logClose() {
        --this._logIndentLevel;
    }
    _logEnabled(message) {
        const spaces = GLTFFileLoader._logSpaces.substring(0, this._logIndentLevel * 2);
        Logger.Log(`${spaces}${message}`);
    }
    _logDisabled(message) { }
    _startPerformanceCounterEnabled(counterName) {
        Tools.StartPerformanceCounter(counterName);
    }
    _startPerformanceCounterDisabled(counterName) { }
    _endPerformanceCounterEnabled(counterName) {
        Tools.EndPerformanceCounter(counterName);
    }
    _endPerformanceCounterDisabled(counterName) { }
}
// ------------------
// End Common options
// ------------------
// ----------------
// Begin V1 options
// ----------------
/**
 * Set this property to false to disable incremental loading which delays the loader from calling the success callback until after loading the meshes and shaders.
 * Textures always loads asynchronously. For example, the success callback can compute the bounding information of the loaded meshes when incremental loading is disabled.
 * Defaults to true.
 * @internal
 */
GLTFFileLoader.IncrementalLoading = true;
/**
 * Set this property to true in order to work with homogeneous coordinates, available with some converters and exporters.
 * Defaults to false. See https://en.wikipedia.org/wiki/Homogeneous_coordinates.
 * @internal
 */
GLTFFileLoader.HomogeneousCoordinates = false;
GLTFFileLoader._logSpaces = "                                ";
RegisterSceneLoaderPlugin(new GLTFFileLoader());

// Caching these dynamic imports gives a surprising perf boost (compared to importing them directly each time).
const LazyAnimationGroupModulePromise = new Lazy(() => __vitePreload(() => import('./animationGroup-CGqF01WM.js'),true              ?__vite__mapDeps([0,1,2,3,4]):void 0,import.meta.url));
const LazyLoaderAnimationModulePromise = new Lazy(() => __vitePreload(() => import('./glTFLoaderAnimation-DNslULmB.js'),true              ?__vite__mapDeps([5,1,2,3,6]):void 0,import.meta.url));
/**
 * Helper class for working with arrays when loading the glTF asset
 */
class ArrayItem {
    /**
     * Gets an item from the given array.
     * @param context The context when loading the asset
     * @param array The array to get the item from
     * @param index The index to the array
     * @returns The array item
     */
    static Get(context, array, index) {
        if (!array || index == undefined || !array[index]) {
            throw new Error(`${context}: Failed to find index (${index})`);
        }
        return array[index];
    }
    /**
     * Gets an item from the given array or returns null if not available.
     * @param array The array to get the item from
     * @param index The index to the array
     * @returns The array item or null
     */
    static TryGet(array, index) {
        if (!array || index == undefined || !array[index]) {
            return null;
        }
        return array[index];
    }
    /**
     * Assign an `index` field to each item of the given array.
     * @param array The array of items
     */
    static Assign(array) {
        if (array) {
            for (let index = 0; index < array.length; index++) {
                array[index].index = index;
            }
        }
    }
}
/** @internal */
function LoadBoundingInfoFromPositionAccessor(accessor) {
    if (accessor.min && accessor.max) {
        const minArray = accessor.min;
        const maxArray = accessor.max;
        const minVector = TmpVectors.Vector3[0].copyFromFloats(minArray[0], minArray[1], minArray[2]);
        const maxVector = TmpVectors.Vector3[1].copyFromFloats(maxArray[0], maxArray[1], maxArray[2]);
        if (accessor.normalized && accessor.componentType !== 5126 /* AccessorComponentType.FLOAT */) {
            let divider = 1;
            switch (accessor.componentType) {
                case 5120 /* AccessorComponentType.BYTE */:
                    divider = 127.0;
                    break;
                case 5121 /* AccessorComponentType.UNSIGNED_BYTE */:
                    divider = 255.0;
                    break;
                case 5122 /* AccessorComponentType.SHORT */:
                    divider = 32767.0;
                    break;
                case 5123 /* AccessorComponentType.UNSIGNED_SHORT */:
                    divider = 65535.0;
                    break;
            }
            const oneOverDivider = 1 / divider;
            minVector.scaleInPlace(oneOverDivider);
            maxVector.scaleInPlace(oneOverDivider);
        }
        return new BoundingInfo(minVector, maxVector);
    }
    return null;
}
/**
 * The glTF 2.0 loader
 */
class GLTFLoader {
    /**
     * Registers a loader extension.
     * @param name The name of the loader extension.
     * @param factory The factory function that creates the loader extension.
     * @deprecated Please use registerGLTFExtension instead.
     */
    static RegisterExtension(name, factory) {
        registerGLTFExtension(name, false, factory);
    }
    /**
     * Unregisters a loader extension.
     * @param name The name of the loader extension.
     * @returns A boolean indicating whether the extension has been unregistered
     * @deprecated Please use unregisterGLTFExtension instead.
     */
    static UnregisterExtension(name) {
        return unregisterGLTFExtension(name);
    }
    /**
     * The object that represents the glTF JSON.
     */
    get gltf() {
        if (!this._gltf) {
            throw new Error("glTF JSON is not available");
        }
        return this._gltf;
    }
    /**
     * The BIN chunk of a binary glTF.
     */
    get bin() {
        return this._bin;
    }
    /**
     * The parent file loader.
     */
    get parent() {
        return this._parent;
    }
    /**
     * The Babylon scene when loading the asset.
     */
    get babylonScene() {
        if (!this._babylonScene) {
            throw new Error("Scene is not available");
        }
        return this._babylonScene;
    }
    /**
     * The root Babylon node when loading the asset.
     */
    get rootBabylonMesh() {
        return this._rootBabylonMesh;
    }
    /**
     * The root url when loading the asset.
     */
    get rootUrl() {
        return this._rootUrl;
    }
    /**
     * @internal
     */
    constructor(parent) {
        /** @internal */
        this._completePromises = new Array();
        /** @internal */
        this._assetContainer = null;
        /** Storage */
        this._babylonLights = [];
        /** @internal */
        this._disableInstancedMesh = 0;
        /** @internal */
        this._allMaterialsDirtyRequired = false;
        /** @internal */
        this._skipStartAnimationStep = false;
        this._extensions = new Array();
        this._disposed = false;
        this._rootUrl = null;
        this._fileName = null;
        this._uniqueRootUrl = null;
        this._bin = null;
        this._rootBabylonMesh = null;
        this._defaultBabylonMaterialData = {};
        this._postSceneLoadActions = new Array();
        this._materialAdapterCache = new WeakMap();
        /** @internal */
        this._pbrMaterialImpl = null;
        this._parent = parent;
    }
    /**
     * Creates or gets a cached material loading adapter with dynamic imports
     * @param material The material to adapt
     * @returns Promise that resolves to the appropriate adapter
     * @internal
     */
    _getOrCreateMaterialAdapter(material) {
        let adapter = this._materialAdapterCache.get(material);
        if (!adapter) {
            if (this._pbrMaterialImpl) {
                adapter = new this._pbrMaterialImpl.adapterClass(material);
            }
            else {
                throw new Error(`Appropriate material adapter class not found`);
            }
            this._materialAdapterCache.set(material, adapter);
        }
        return adapter;
    }
    /** @internal */
    dispose() {
        if (this._disposed) {
            return;
        }
        this._disposed = true;
        this._completePromises.length = 0;
        this._extensions.forEach((extension) => extension.dispose && extension.dispose());
        this._extensions.length = 0;
        this._gltf = null; // TODO
        this._bin = null;
        this._babylonScene = null; // TODO
        this._rootBabylonMesh = null;
        this._defaultBabylonMaterialData = {};
        this._postSceneLoadActions.length = 0;
        this._parent.dispose();
    }
    /**
     * @internal
     */
    async importMeshAsync(meshesNames, scene, container, data, rootUrl, onProgress, fileName = "") {
        // eslint-disable-next-line github/no-then
        return await Promise.resolve().then(async () => {
            this._babylonScene = scene;
            this._assetContainer = container;
            this._loadData(data);
            let nodes = null;
            if (meshesNames) {
                const nodeMap = {};
                if (this._gltf.nodes) {
                    for (const node of this._gltf.nodes) {
                        if (node.name) {
                            nodeMap[node.name] = node.index;
                        }
                    }
                }
                const names = meshesNames instanceof Array ? meshesNames : [meshesNames];
                nodes = names.map((name) => {
                    const node = nodeMap[name];
                    if (node === undefined) {
                        throw new Error(`Failed to find node '${name}'`);
                    }
                    return node;
                });
            }
            return await this._loadAsync(rootUrl, fileName, nodes, () => {
                return {
                    meshes: this._getMeshes(),
                    particleSystems: [],
                    skeletons: this._getSkeletons(),
                    animationGroups: this._getAnimationGroups(),
                    lights: this._babylonLights,
                    transformNodes: this._getTransformNodes(),
                    geometries: this._getGeometries(),
                    spriteManagers: [],
                };
            });
        });
    }
    /**
     * @internal
     */
    async loadAsync(scene, data, rootUrl, onProgress, fileName = "") {
        this._babylonScene = scene;
        this._loadData(data);
        return await this._loadAsync(rootUrl, fileName, null, () => undefined);
    }
    async _loadAsync(rootUrl, fileName, nodes, resultFunc) {
        return await Promise.resolve()
            .then(async () => {
            this._rootUrl = rootUrl;
            this._uniqueRootUrl = !rootUrl.startsWith("file:") && fileName ? rootUrl : `${rootUrl}${Date.now()}/`;
            this._fileName = fileName;
            this._allMaterialsDirtyRequired = false;
            await this._loadExtensionsAsync();
            // NOTE: Explicitly check _pbrMaterialImpl for null as a value of false means don't use PBR materials at all.
            if (!this.parent.skipMaterials && this._pbrMaterialImpl == null) {
                if (this.parent.useOpenPBR || this.isExtensionUsed("KHR_materials_openpbr")) {
                    this._pbrMaterialImpl = {
                        materialClass: (await __vitePreload(async () => { const {OpenPBRMaterial} = await import('./openpbrMaterial-L3-DYNd8.js');return { OpenPBRMaterial }},true              ?__vite__mapDeps([7,2,3,1]):void 0,import.meta.url)).OpenPBRMaterial,
                        adapterClass: (await __vitePreload(async () => { const {OpenPBRMaterialLoadingAdapter} = await import('./openpbrMaterialLoadingAdapter-Br2hIutA.js');return { OpenPBRMaterialLoadingAdapter }},true              ?[]:void 0,import.meta.url)).OpenPBRMaterialLoadingAdapter,
                    };
                }
                else {
                    this._pbrMaterialImpl = {
                        materialClass: (await __vitePreload(async () => { const {PBRMaterial} = await import('./index-CxR0Yf_7.js').then(n => n.cw);return { PBRMaterial }},true              ?__vite__mapDeps([1,2,3]):void 0,import.meta.url)).PBRMaterial,
                        adapterClass: (await __vitePreload(async () => { const {PBRMaterialLoadingAdapter} = await import('./pbrMaterialLoadingAdapter-vcu2PXuo.js');return { PBRMaterialLoadingAdapter }},true              ?__vite__mapDeps([8,1,2,3]):void 0,import.meta.url)).PBRMaterialLoadingAdapter,
                    };
                }
            }
            const loadingToReadyCounterName = `${GLTFLoaderState[GLTFLoaderState.LOADING]} => ${GLTFLoaderState[GLTFLoaderState.READY]}`;
            const loadingToCompleteCounterName = `${GLTFLoaderState[GLTFLoaderState.LOADING]} => ${GLTFLoaderState[GLTFLoaderState.COMPLETE]}`;
            this._parent._startPerformanceCounter(loadingToReadyCounterName);
            this._parent._startPerformanceCounter(loadingToCompleteCounterName);
            this._parent._setState(GLTFLoaderState.LOADING);
            this._extensionsOnLoading();
            const promises = new Array();
            // Block the marking of materials dirty until the scene is loaded.
            const oldBlockMaterialDirtyMechanism = this._babylonScene.blockMaterialDirtyMechanism;
            this._babylonScene.blockMaterialDirtyMechanism = true;
            if (!this.parent.loadOnlyMaterials) {
                if (nodes) {
                    promises.push(this.loadSceneAsync("/nodes", { nodes: nodes, index: -1 }));
                }
                else if (this._gltf.scene != undefined || (this._gltf.scenes && this._gltf.scenes[0])) {
                    const scene = ArrayItem.Get(`/scene`, this._gltf.scenes, this._gltf.scene || 0);
                    promises.push(this.loadSceneAsync(`/scenes/${scene.index}`, scene));
                }
            }
            if (!this.parent.skipMaterials && this.parent.loadAllMaterials && this._gltf.materials) {
                for (let m = 0; m < this._gltf.materials.length; ++m) {
                    const material = this._gltf.materials[m];
                    const context = "/materials/" + m;
                    const babylonDrawMode = Material.TriangleFillMode;
                    promises.push(this._loadMaterialAsync(context, material, null, babylonDrawMode, () => { }));
                }
            }
            // Restore the blocking of material dirty.
            if (this._allMaterialsDirtyRequired) {
                // This can happen if we add a light for instance as it will impact the whole scene.
                // This automatically resets everything if needed.
                this._babylonScene.blockMaterialDirtyMechanism = oldBlockMaterialDirtyMechanism;
            }
            else {
                // By default a newly created material is dirty so there is no need to flag the full scene as dirty.
                // For perf reasons, we then bypass blockMaterialDirtyMechanism as this would "dirty" the entire scene.
                this._babylonScene._forceBlockMaterialDirtyMechanism(oldBlockMaterialDirtyMechanism);
            }
            if (this._parent.compileMaterials) {
                promises.push(this._compileMaterialsAsync());
            }
            if (this._parent.compileShadowGenerators) {
                promises.push(this._compileShadowGeneratorsAsync());
            }
            const resultPromise = Promise.all(promises).then(() => {
                if (this._rootBabylonMesh && this._rootBabylonMesh !== this._parent.customRootNode) {
                    this._rootBabylonMesh.setEnabled(true);
                }
                // Making sure we enable enough lights to have all lights together
                for (const material of this._babylonScene.materials) {
                    const mat = material;
                    if (mat.maxSimultaneousLights !== undefined) {
                        mat.maxSimultaneousLights = Math.max(mat.maxSimultaneousLights, this._babylonScene.lights.length);
                    }
                }
                this._extensionsOnReady();
                this._parent._setState(GLTFLoaderState.READY);
                if (!this._skipStartAnimationStep) {
                    this._startAnimations();
                }
                return resultFunc();
            });
            return await resultPromise.then((result) => {
                this._parent._endPerformanceCounter(loadingToReadyCounterName);
                Tools.SetImmediate(() => {
                    if (!this._disposed) {
                        Promise.all(this._completePromises).then(() => {
                            this._parent._endPerformanceCounter(loadingToCompleteCounterName);
                            this._parent._setState(GLTFLoaderState.COMPLETE);
                            this._parent.onCompleteObservable.notifyObservers(undefined);
                            this._parent.onCompleteObservable.clear();
                            this.dispose();
                        }, (error) => {
                            this._parent.onErrorObservable.notifyObservers(error);
                            this._parent.onErrorObservable.clear();
                            this.dispose();
                        });
                    }
                });
                return result;
            });
        })
            .catch((error) => {
            if (!this._disposed) {
                this._parent.onErrorObservable.notifyObservers(error);
                this._parent.onErrorObservable.clear();
                this.dispose();
            }
            throw error;
        });
    }
    _loadData(data) {
        this._gltf = data.json;
        this._setupData();
        if (data.bin) {
            const buffers = this._gltf.buffers;
            if (buffers && buffers[0] && !buffers[0].uri) {
                const binaryBuffer = buffers[0];
                if (binaryBuffer.byteLength < data.bin.byteLength - 3 || binaryBuffer.byteLength > data.bin.byteLength) {
                    Logger.Warn(`Binary buffer length (${binaryBuffer.byteLength}) from JSON does not match chunk length (${data.bin.byteLength})`);
                }
                this._bin = data.bin;
            }
            else {
                Logger.Warn("Unexpected BIN chunk");
            }
        }
    }
    _setupData() {
        ArrayItem.Assign(this._gltf.accessors);
        ArrayItem.Assign(this._gltf.animations);
        ArrayItem.Assign(this._gltf.buffers);
        ArrayItem.Assign(this._gltf.bufferViews);
        ArrayItem.Assign(this._gltf.cameras);
        ArrayItem.Assign(this._gltf.images);
        ArrayItem.Assign(this._gltf.materials);
        ArrayItem.Assign(this._gltf.meshes);
        ArrayItem.Assign(this._gltf.nodes);
        ArrayItem.Assign(this._gltf.samplers);
        ArrayItem.Assign(this._gltf.scenes);
        ArrayItem.Assign(this._gltf.skins);
        ArrayItem.Assign(this._gltf.textures);
        if (this._gltf.nodes) {
            const nodeParents = {};
            for (const node of this._gltf.nodes) {
                if (node.children) {
                    for (const index of node.children) {
                        nodeParents[index] = node.index;
                    }
                }
            }
            const rootNode = this._createRootNode();
            for (const node of this._gltf.nodes) {
                const parentIndex = nodeParents[node.index];
                node.parent = parentIndex === undefined ? rootNode : this._gltf.nodes[parentIndex];
            }
        }
    }
    async _loadExtensionsAsync() {
        const extensionPromises = [];
        registeredGLTFExtensions.forEach((registeredExtension, name) => {
            // Don't load explicitly disabled extensions.
            if (this.parent.extensionOptions[name]?.enabled === false) {
                // But warn if the disabled extension is used by the model.
                if (registeredExtension.isGLTFExtension && this.isExtensionUsed(name)) {
                    Logger.Warn(`Extension ${name} is used but has been explicitly disabled.`);
                }
            }
            // Load loader extensions that are not a glTF extension, as well as extensions that are glTF extensions and are used by the model.
            else if (!registeredExtension.isGLTFExtension || this.isExtensionUsed(name)) {
                extensionPromises.push((async () => {
                    const extension = await registeredExtension.factory(this);
                    if (extension.name !== name) {
                        Logger.Warn(`The name of the glTF loader extension instance does not match the registered name: ${extension.name} !== ${name}`);
                    }
                    this._parent.onExtensionLoadedObservable.notifyObservers(extension);
                    return extension;
                })());
            }
        });
        this._extensions.push(...(await Promise.all(extensionPromises)));
        this._extensions.sort((a, b) => (a.order || Number.MAX_VALUE) - (b.order || Number.MAX_VALUE));
        this._parent.onExtensionLoadedObservable.clear();
        if (this._gltf.extensionsRequired) {
            for (const name of this._gltf.extensionsRequired) {
                const available = this._extensions.some((extension) => extension.name === name && extension.enabled);
                if (!available) {
                    if (this.parent.extensionOptions[name]?.enabled === false) {
                        throw new Error(`Required extension ${name} is disabled`);
                    }
                    throw new Error(`Required extension ${name} is not available`);
                }
            }
        }
    }
    _createRootNode() {
        if (this._parent.customRootNode !== undefined) {
            this._rootBabylonMesh = this._parent.customRootNode;
            return {
                // eslint-disable-next-line @typescript-eslint/naming-convention
                _babylonTransformNode: this._rootBabylonMesh === null ? undefined : this._rootBabylonMesh,
                index: -1,
            };
        }
        this._babylonScene._blockEntityCollection = !!this._assetContainer;
        const rootMesh = new Mesh("__root__", this._babylonScene);
        this._rootBabylonMesh = rootMesh;
        this._rootBabylonMesh._parentContainer = this._assetContainer;
        this._babylonScene._blockEntityCollection = false;
        this._rootBabylonMesh.setEnabled(false);
        const rootNode = {
            // eslint-disable-next-line @typescript-eslint/naming-convention
            _babylonTransformNode: this._rootBabylonMesh,
            index: -1,
        };
        switch (this._parent.coordinateSystemMode) {
            case GLTFLoaderCoordinateSystemMode.AUTO: {
                if (!this._babylonScene.useRightHandedSystem) {
                    rootNode.rotation = [0, 1, 0, 0];
                    rootNode.scale = [1, 1, -1];
                    GLTFLoader._LoadTransform(rootNode, this._rootBabylonMesh);
                }
                break;
            }
            case GLTFLoaderCoordinateSystemMode.FORCE_RIGHT_HANDED: {
                this._babylonScene.useRightHandedSystem = true;
                break;
            }
            default: {
                throw new Error(`Invalid coordinate system mode (${this._parent.coordinateSystemMode})`);
            }
        }
        this._parent.onMeshLoadedObservable.notifyObservers(rootMesh);
        return rootNode;
    }
    /**
     * Loads a glTF scene.
     * @param context The context when loading the asset
     * @param scene The glTF scene property
     * @returns A promise that resolves when the load is complete
     */
    // eslint-disable-next-line no-restricted-syntax, @typescript-eslint/promise-function-async
    loadSceneAsync(context, scene) {
        const extensionPromise = this._extensionsLoadSceneAsync(context, scene);
        if (extensionPromise) {
            return extensionPromise;
        }
        const promises = new Array();
        this.logOpen(`${context} ${scene.name || ""}`);
        if (scene.nodes) {
            for (const index of scene.nodes) {
                const node = ArrayItem.Get(`${context}/nodes/${index}`, this._gltf.nodes, index);
                promises.push(this.loadNodeAsync(`/nodes/${node.index}`, node, (babylonMesh) => {
                    babylonMesh.parent = this._rootBabylonMesh;
                }));
            }
        }
        for (const action of this._postSceneLoadActions) {
            action();
        }
        promises.push(this._loadAnimationsAsync());
        this.logClose();
        return Promise.all(promises).then(() => { });
    }
    _forEachPrimitive(node, callback) {
        if (node._primitiveBabylonMeshes) {
            for (const babylonMesh of node._primitiveBabylonMeshes) {
                callback(babylonMesh);
            }
        }
    }
    _getGeometries() {
        const geometries = [];
        const nodes = this._gltf.nodes;
        if (nodes) {
            for (const node of nodes) {
                this._forEachPrimitive(node, (babylonMesh) => {
                    const geometry = babylonMesh.geometry;
                    if (geometry && geometries.indexOf(geometry) === -1) {
                        geometries.push(geometry);
                    }
                });
            }
        }
        return geometries;
    }
    _getMeshes() {
        const meshes = [];
        // Root mesh is always first, if available.
        if (this._rootBabylonMesh instanceof AbstractMesh) {
            meshes.push(this._rootBabylonMesh);
        }
        const nodes = this._gltf.nodes;
        if (nodes) {
            for (const node of nodes) {
                this._forEachPrimitive(node, (babylonMesh) => {
                    meshes.push(babylonMesh);
                });
            }
        }
        return meshes;
    }
    _getTransformNodes() {
        const transformNodes = [];
        const nodes = this._gltf.nodes;
        if (nodes) {
            for (const node of nodes) {
                if (node._babylonTransformNode && node._babylonTransformNode.getClassName() === "TransformNode") {
                    transformNodes.push(node._babylonTransformNode);
                }
                if (node._babylonTransformNodeForSkin) {
                    transformNodes.push(node._babylonTransformNodeForSkin);
                }
            }
        }
        return transformNodes;
    }
    _getSkeletons() {
        const skeletons = [];
        const skins = this._gltf.skins;
        if (skins) {
            for (const skin of skins) {
                if (skin._data) {
                    skeletons.push(skin._data.babylonSkeleton);
                }
            }
        }
        return skeletons;
    }
    _getAnimationGroups() {
        const animationGroups = [];
        const animations = this._gltf.animations;
        if (animations) {
            for (const animation of animations) {
                if (animation._babylonAnimationGroup) {
                    animationGroups.push(animation._babylonAnimationGroup);
                }
            }
        }
        return animationGroups;
    }
    _startAnimations() {
        switch (this._parent.animationStartMode) {
            case GLTFLoaderAnimationStartMode.NONE: {
                // do nothing
                break;
            }
            case GLTFLoaderAnimationStartMode.FIRST: {
                const babylonAnimationGroups = this._getAnimationGroups();
                if (babylonAnimationGroups.length !== 0) {
                    babylonAnimationGroups[0].start(true);
                }
                break;
            }
            case GLTFLoaderAnimationStartMode.ALL: {
                const babylonAnimationGroups = this._getAnimationGroups();
                for (const babylonAnimationGroup of babylonAnimationGroups) {
                    babylonAnimationGroup.start(true);
                }
                break;
            }
            default: {
                Logger.Error(`Invalid animation start mode (${this._parent.animationStartMode})`);
                return;
            }
        }
    }
    /**
     * Loads a glTF node.
     * @param context The context when loading the asset
     * @param node The glTF node property
     * @param assign A function called synchronously after parsing the glTF properties
     * @returns A promise that resolves with the loaded Babylon mesh when the load is complete
     */
    // eslint-disable-next-line @typescript-eslint/promise-function-async, no-restricted-syntax
    loadNodeAsync(context, node, assign = () => { }) {
        const extensionPromise = this._extensionsLoadNodeAsync(context, node, assign);
        if (extensionPromise) {
            return extensionPromise;
        }
        if (node._babylonTransformNode) {
            throw new Error(`${context}: Invalid recursive node hierarchy`);
        }
        const promises = new Array();
        this.logOpen(`${context} ${node.name || ""}`);
        const loadNode = (babylonTransformNode) => {
            GLTFLoader.AddPointerMetadata(babylonTransformNode, context);
            GLTFLoader._LoadTransform(node, babylonTransformNode);
            if (node.camera != undefined) {
                const camera = ArrayItem.Get(`${context}/camera`, this._gltf.cameras, node.camera);
                promises.push(this.loadCameraAsync(`/cameras/${camera.index}`, camera, (babylonCamera) => {
                    babylonCamera.parent = babylonTransformNode;
                    if (!this._babylonScene.useRightHandedSystem) {
                        babylonTransformNode.scaling.x = -1; // Cancelling root node scaling for handedness so the view matrix does not end up flipped.
                    }
                }));
            }
            if (node.children) {
                for (const index of node.children) {
                    const childNode = ArrayItem.Get(`${context}/children/${index}`, this._gltf.nodes, index);
                    promises.push(this.loadNodeAsync(`/nodes/${childNode.index}`, childNode, (childBabylonMesh) => {
                        childBabylonMesh.parent = babylonTransformNode;
                    }));
                }
            }
            assign(babylonTransformNode);
        };
        const hasMesh = node.mesh != undefined;
        const hasSkin = this._parent.loadSkins && node.skin != undefined;
        if (!hasMesh || hasSkin) {
            const nodeName = node.name || `node${node.index}`;
            this._babylonScene._blockEntityCollection = !!this._assetContainer;
            const transformNode = new TransformNode(nodeName, this._babylonScene);
            transformNode._parentContainer = this._assetContainer;
            this._babylonScene._blockEntityCollection = false;
            if (node.mesh == undefined) {
                node._babylonTransformNode = transformNode;
            }
            else {
                node._babylonTransformNodeForSkin = transformNode;
            }
            loadNode(transformNode);
        }
        if (hasMesh) {
            if (hasSkin) {
                // See https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#skins (second implementation note)
                // This code path will place the skinned mesh as a sibling of the skeleton root node without loading the
                // transform, which effectively ignores the transform of the skinned mesh, as per spec.
                const mesh = ArrayItem.Get(`${context}/mesh`, this._gltf.meshes, node.mesh);
                promises.push(this._loadMeshAsync(`/meshes/${mesh.index}`, node, mesh, (babylonTransformNode) => {
                    const babylonTransformNodeForSkin = node._babylonTransformNodeForSkin;
                    // Merge the metadata from the skin node to the skinned mesh in case a loader extension added metadata.
                    babylonTransformNode.metadata = deepMerge(babylonTransformNodeForSkin.metadata, babylonTransformNode.metadata || {});
                    const skin = ArrayItem.Get(`${context}/skin`, this._gltf.skins, node.skin);
                    promises.push(this._loadSkinAsync(`/skins/${skin.index}`, node, skin, (babylonSkeleton) => {
                        this._forEachPrimitive(node, (babylonMesh) => {
                            babylonMesh.skeleton = babylonSkeleton;
                        });
                        // Wait until all the nodes are parented before parenting the skinned mesh.
                        this._postSceneLoadActions.push(() => {
                            if (skin.skeleton != undefined) {
                                // Place the skinned mesh node as a sibling of the skeleton root node.
                                // Handle special case when the parent of the skeleton root is the skinned mesh.
                                const parentNode = ArrayItem.Get(`/skins/${skin.index}/skeleton`, this._gltf.nodes, skin.skeleton).parent;
                                if (node.index === parentNode.index) {
                                    babylonTransformNode.parent = babylonTransformNodeForSkin.parent;
                                }
                                else {
                                    babylonTransformNode.parent = parentNode._babylonTransformNode;
                                }
                            }
                            else {
                                babylonTransformNode.parent = this._rootBabylonMesh;
                            }
                            this._parent.onSkinLoadedObservable.notifyObservers({ node: babylonTransformNodeForSkin, skinnedNode: babylonTransformNode });
                        });
                    }));
                }));
            }
            else {
                const mesh = ArrayItem.Get(`${context}/mesh`, this._gltf.meshes, node.mesh);
                promises.push(this._loadMeshAsync(`/meshes/${mesh.index}`, node, mesh, loadNode));
            }
        }
        this.logClose();
        return Promise.all(promises).then(() => {
            this._forEachPrimitive(node, (babylonMesh) => {
                const asMesh = babylonMesh;
                if (!asMesh.isAnInstance && asMesh.geometry && asMesh.geometry.useBoundingInfoFromGeometry) {
                    // simply apply the world matrices to the bounding info - the extends are already ok
                    babylonMesh._updateBoundingInfo();
                }
                else {
                    babylonMesh.refreshBoundingInfo(true, true);
                }
            });
            return node._babylonTransformNode;
        });
    }
    // eslint-disable-next-line @typescript-eslint/promise-function-async, no-restricted-syntax
    _loadMeshAsync(context, node, mesh, assign) {
        const primitives = mesh.primitives;
        if (!primitives || !primitives.length) {
            throw new Error(`${context}: Primitives are missing`);
        }
        if (primitives[0].index == undefined) {
            ArrayItem.Assign(primitives);
        }
        const promises = new Array();
        this.logOpen(`${context} ${mesh.name || ""}`);
        const name = node.name || `node${node.index}`;
        if (primitives.length === 1) {
            const primitive = mesh.primitives[0];
            promises.push(this._loadMeshPrimitiveAsync(`${context}/primitives/${primitive.index}`, name, node, mesh, primitive, (babylonMesh) => {
                node._babylonTransformNode = babylonMesh;
                node._primitiveBabylonMeshes = [babylonMesh];
            }));
        }
        else {
            this._babylonScene._blockEntityCollection = !!this._assetContainer;
            node._babylonTransformNode = new TransformNode(name, this._babylonScene);
            node._babylonTransformNode._parentContainer = this._assetContainer;
            this._babylonScene._blockEntityCollection = false;
            node._primitiveBabylonMeshes = [];
            for (const primitive of primitives) {
                promises.push(this._loadMeshPrimitiveAsync(`${context}/primitives/${primitive.index}`, `${name}_primitive${primitive.index}`, node, mesh, primitive, (babylonMesh) => {
                    babylonMesh.parent = node._babylonTransformNode;
                    node._primitiveBabylonMeshes.push(babylonMesh);
                }));
            }
        }
        assign(node._babylonTransformNode);
        this.logClose();
        return Promise.all(promises).then(() => {
            return node._babylonTransformNode;
        });
    }
    /**
     * @internal Define this method to modify the default behavior when loading data for mesh primitives.
     * @param context The context when loading the asset
     * @param name The mesh name when loading the asset
     * @param node The glTF node when loading the asset
     * @param mesh The glTF mesh when loading the asset
     * @param primitive The glTF mesh primitive property
     * @param assign A function called synchronously after parsing the glTF properties
     * @returns A promise that resolves with the loaded mesh when the load is complete or null if not handled
     */
    // eslint-disable-next-line @typescript-eslint/promise-function-async, no-restricted-syntax
    _loadMeshPrimitiveAsync(context, name, node, mesh, primitive, assign) {
        const extensionPromise = this._extensionsLoadMeshPrimitiveAsync(context, name, node, mesh, primitive, assign);
        if (extensionPromise) {
            return extensionPromise;
        }
        this.logOpen(`${context}`);
        const shouldInstance = this._disableInstancedMesh === 0 && this._parent.createInstances && node.skin == undefined && !mesh.primitives[0].targets;
        let babylonAbstractMesh;
        let promise;
        if (shouldInstance && primitive._instanceData) {
            this._babylonScene._blockEntityCollection = !!this._assetContainer;
            babylonAbstractMesh = primitive._instanceData.babylonSourceMesh.createInstance(name);
            babylonAbstractMesh._parentContainer = this._assetContainer;
            this._babylonScene._blockEntityCollection = false;
            promise = primitive._instanceData.promise;
        }
        else {
            const promises = new Array();
            this._babylonScene._blockEntityCollection = !!this._assetContainer;
            const babylonMesh = new Mesh(name, this._babylonScene);
            babylonMesh._parentContainer = this._assetContainer;
            this._babylonScene._blockEntityCollection = false;
            babylonMesh.sideOrientation = this._babylonScene.useRightHandedSystem ? Material.CounterClockWiseSideOrientation : Material.ClockWiseSideOrientation;
            this._createMorphTargets(context, node, mesh, primitive, babylonMesh);
            promises.push(this._loadVertexDataAsync(context, primitive, babylonMesh).then(async (babylonGeometry) => {
                return await this._loadMorphTargetsAsync(context, primitive, babylonMesh, babylonGeometry).then(() => {
                    if (this._disposed) {
                        return;
                    }
                    this._babylonScene._blockEntityCollection = !!this._assetContainer;
                    babylonGeometry.applyToMesh(babylonMesh);
                    babylonGeometry._parentContainer = this._assetContainer;
                    this._babylonScene._blockEntityCollection = false;
                });
            }));
            if (!this.parent.skipMaterials) {
                const babylonDrawMode = GLTFLoader._GetDrawMode(context, primitive.mode);
                if (primitive.material == undefined) {
                    let babylonMaterial = this._defaultBabylonMaterialData[babylonDrawMode];
                    if (!babylonMaterial) {
                        babylonMaterial = this._createDefaultMaterial("__GLTFLoader._default", babylonDrawMode);
                        this._parent.onMaterialLoadedObservable.notifyObservers(babylonMaterial);
                        this._defaultBabylonMaterialData[babylonDrawMode] = babylonMaterial;
                    }
                    babylonMesh.material = babylonMaterial;
                }
                else {
                    const material = ArrayItem.Get(`${context}/material`, this._gltf.materials, primitive.material);
                    promises.push(this._loadMaterialAsync(`/materials/${material.index}`, material, babylonMesh, babylonDrawMode, (babylonMaterial) => {
                        babylonMesh.material = babylonMaterial;
                    }));
                }
            }
            promise = Promise.all(promises);
            if (shouldInstance) {
                primitive._instanceData = {
                    babylonSourceMesh: babylonMesh,
                    promise: promise,
                };
            }
            babylonAbstractMesh = babylonMesh;
        }
        GLTFLoader.AddPointerMetadata(babylonAbstractMesh, context);
        this._parent.onMeshLoadedObservable.notifyObservers(babylonAbstractMesh);
        assign(babylonAbstractMesh);
        this.logClose();
        return promise.then(() => {
            return babylonAbstractMesh;
        });
    }
    // eslint-disable-next-line @typescript-eslint/promise-function-async, no-restricted-syntax
    _loadVertexDataAsync(context, primitive, babylonMesh) {
        const extensionPromise = this._extensionsLoadVertexDataAsync(context, primitive, babylonMesh);
        if (extensionPromise) {
            return extensionPromise;
        }
        const attributes = primitive.attributes;
        if (!attributes) {
            throw new Error(`${context}: Attributes are missing`);
        }
        const promises = new Array();
        const babylonGeometry = new Geometry(babylonMesh.name, this._babylonScene);
        if (primitive.indices == undefined) {
            babylonMesh.isUnIndexed = true;
        }
        else {
            const accessor = ArrayItem.Get(`${context}/indices`, this._gltf.accessors, primitive.indices);
            promises.push(this._loadIndicesAccessorAsync(`/accessors/${accessor.index}`, accessor).then((data) => {
                babylonGeometry.setIndices(data);
            }));
        }
        const loadAttribute = (name, kind, callback) => {
            if (attributes[name] == undefined) {
                return;
            }
            babylonMesh._delayInfo = babylonMesh._delayInfo || [];
            if (babylonMesh._delayInfo.indexOf(kind) === -1) {
                babylonMesh._delayInfo.push(kind);
            }
            const accessor = ArrayItem.Get(`${context}/attributes/${name}`, this._gltf.accessors, attributes[name]);
            promises.push(this._loadVertexAccessorAsync(`/accessors/${accessor.index}`, accessor, kind).then((babylonVertexBuffer) => {
                if (babylonVertexBuffer.getKind() === VertexBuffer.PositionKind && !this.parent.alwaysComputeBoundingBox && !babylonMesh.skeleton) {
                    const babylonBoundingInfo = LoadBoundingInfoFromPositionAccessor(accessor);
                    if (babylonBoundingInfo) {
                        babylonGeometry._boundingInfo = babylonBoundingInfo;
                        babylonGeometry.useBoundingInfoFromGeometry = true;
                    }
                }
                babylonGeometry.setVerticesBuffer(babylonVertexBuffer, accessor.count);
            }));
            if (kind == VertexBuffer.MatricesIndicesExtraKind) {
                babylonMesh.numBoneInfluencers = 8;
            }
            if (callback) {
                callback(accessor);
            }
        };
        loadAttribute("POSITION", VertexBuffer.PositionKind);
        loadAttribute("NORMAL", VertexBuffer.NormalKind);
        loadAttribute("TANGENT", VertexBuffer.TangentKind);
        loadAttribute("TEXCOORD_0", VertexBuffer.UVKind);
        loadAttribute("TEXCOORD_1", VertexBuffer.UV2Kind);
        loadAttribute("TEXCOORD_2", VertexBuffer.UV3Kind);
        loadAttribute("TEXCOORD_3", VertexBuffer.UV4Kind);
        loadAttribute("TEXCOORD_4", VertexBuffer.UV5Kind);
        loadAttribute("TEXCOORD_5", VertexBuffer.UV6Kind);
        loadAttribute("JOINTS_0", VertexBuffer.MatricesIndicesKind);
        loadAttribute("WEIGHTS_0", VertexBuffer.MatricesWeightsKind);
        loadAttribute("JOINTS_1", VertexBuffer.MatricesIndicesExtraKind);
        loadAttribute("WEIGHTS_1", VertexBuffer.MatricesWeightsExtraKind);
        loadAttribute("COLOR_0", VertexBuffer.ColorKind, (accessor) => {
            if (accessor.type === "VEC4" /* AccessorType.VEC4 */) {
                babylonMesh.hasVertexAlpha = true;
            }
        });
        return Promise.all(promises).then(() => {
            return babylonGeometry;
        });
    }
    _createMorphTargets(context, node, mesh, primitive, babylonMesh) {
        if (!primitive.targets || !this._parent.loadMorphTargets) {
            return;
        }
        if (node._numMorphTargets == undefined) {
            node._numMorphTargets = primitive.targets.length;
        }
        else if (primitive.targets.length !== node._numMorphTargets) {
            throw new Error(`${context}: Primitives do not have the same number of targets`);
        }
        const targetNames = mesh.extras ? mesh.extras.targetNames : null;
        this._babylonScene._blockEntityCollection = !!this._assetContainer;
        babylonMesh.morphTargetManager = new MorphTargetManager(this._babylonScene);
        babylonMesh.morphTargetManager._parentContainer = this._assetContainer;
        this._babylonScene._blockEntityCollection = false;
        babylonMesh.morphTargetManager.areUpdatesFrozen = true;
        for (let index = 0; index < primitive.targets.length; index++) {
            const weight = node.weights ? node.weights[index] : mesh.weights ? mesh.weights[index] : 0;
            const name = targetNames ? targetNames[index] : `morphTarget${index}`;
            babylonMesh.morphTargetManager.addTarget(new MorphTarget(name, weight, babylonMesh.getScene()));
            // TODO: tell the target whether it has positions, normals, tangents
        }
    }
    // eslint-disable-next-line @typescript-eslint/promise-function-async, no-restricted-syntax
    _loadMorphTargetsAsync(context, primitive, babylonMesh, babylonGeometry) {
        if (!primitive.targets || !this._parent.loadMorphTargets) {
            return Promise.resolve();
        }
        const promises = new Array();
        const morphTargetManager = babylonMesh.morphTargetManager;
        for (let index = 0; index < morphTargetManager.numTargets; index++) {
            const babylonMorphTarget = morphTargetManager.getTarget(index);
            promises.push(this._loadMorphTargetVertexDataAsync(`${context}/targets/${index}`, babylonGeometry, primitive.targets[index], babylonMorphTarget));
        }
        return Promise.all(promises).then(() => {
            morphTargetManager.areUpdatesFrozen = false;
        });
    }
    async _loadMorphTargetVertexDataAsync(context, babylonGeometry, attributes, babylonMorphTarget) {
        const promises = new Array();
        const loadAttribute = (attribute, kind, setData) => {
            if (attributes[attribute] == undefined) {
                return;
            }
            const babylonVertexBuffer = babylonGeometry.getVertexBuffer(kind);
            if (!babylonVertexBuffer) {
                return;
            }
            const accessor = ArrayItem.Get(`${context}/${attribute}`, this._gltf.accessors, attributes[attribute]);
            promises.push(this._loadFloatAccessorAsync(`/accessors/${accessor.index}`, accessor).then((data) => {
                setData(babylonVertexBuffer, data);
            }));
        };
        loadAttribute("POSITION", VertexBuffer.PositionKind, (babylonVertexBuffer, data) => {
            const positions = new Float32Array(data.length);
            babylonVertexBuffer.forEach(data.length, (value, index) => {
                positions[index] = data[index] + value;
            });
            babylonMorphTarget.setPositions(positions);
        });
        loadAttribute("NORMAL", VertexBuffer.NormalKind, (babylonVertexBuffer, data) => {
            const normals = new Float32Array(data.length);
            babylonVertexBuffer.forEach(normals.length, (value, index) => {
                normals[index] = data[index] + value;
            });
            babylonMorphTarget.setNormals(normals);
        });
        loadAttribute("TANGENT", VertexBuffer.TangentKind, (babylonVertexBuffer, data) => {
            const tangents = new Float32Array((data.length / 3) * 4);
            let dataIndex = 0;
            babylonVertexBuffer.forEach((data.length / 3) * 4, (value, index) => {
                // Tangent data for morph targets is stored as xyz delta.
                // The vertexData.tangent is stored as xyzw.
                // So we need to skip every fourth vertexData.tangent.
                if ((index + 1) % 4 !== 0) {
                    tangents[dataIndex] = data[dataIndex] + value;
                    dataIndex++;
                }
            });
            babylonMorphTarget.setTangents(tangents);
        });
        loadAttribute("TEXCOORD_0", VertexBuffer.UVKind, (babylonVertexBuffer, data) => {
            const uvs = new Float32Array(data.length);
            babylonVertexBuffer.forEach(data.length, (value, index) => {
                uvs[index] = data[index] + value;
            });
            babylonMorphTarget.setUVs(uvs);
        });
        loadAttribute("TEXCOORD_1", VertexBuffer.UV2Kind, (babylonVertexBuffer, data) => {
            const uvs = new Float32Array(data.length);
            babylonVertexBuffer.forEach(data.length, (value, index) => {
                uvs[index] = data[index] + value;
            });
            babylonMorphTarget.setUV2s(uvs);
        });
        loadAttribute("COLOR_0", VertexBuffer.ColorKind, (babylonVertexBuffer, data) => {
            let colors = null;
            const componentSize = babylonVertexBuffer.getSize();
            if (componentSize === 3) {
                colors = new Float32Array((data.length / 3) * 4);
                babylonVertexBuffer.forEach(data.length, (value, index) => {
                    const pixid = Math.floor(index / 3);
                    const channel = index % 3;
                    colors[4 * pixid + channel] = data[3 * pixid + channel] + value;
                });
                for (let i = 0; i < data.length / 3; ++i) {
                    colors[4 * i + 3] = 1;
                }
            }
            else if (componentSize === 4) {
                colors = new Float32Array(data.length);
                babylonVertexBuffer.forEach(data.length, (value, index) => {
                    colors[index] = data[index] + value;
                });
            }
            else {
                throw new Error(`${context}: Invalid number of components (${componentSize}) for COLOR_0 attribute`);
            }
            babylonMorphTarget.setColors(colors);
        });
        return await Promise.all(promises).then(() => { });
    }
    static _LoadTransform(node, babylonNode) {
        // Ignore the TRS of skinned nodes.
        // See https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#skins (second implementation note)
        if (node.skin != undefined) {
            return;
        }
        let position = Vector3.Zero();
        let rotation = Quaternion.Identity();
        let scaling = Vector3.One();
        if (node.matrix) {
            const matrix = Matrix.FromArray(node.matrix);
            matrix.decompose(scaling, rotation, position);
        }
        else {
            if (node.translation) {
                position = Vector3.FromArray(node.translation);
            }
            if (node.rotation) {
                rotation = Quaternion.FromArray(node.rotation);
            }
            if (node.scale) {
                scaling = Vector3.FromArray(node.scale);
            }
        }
        babylonNode.position = position;
        babylonNode.rotationQuaternion = rotation;
        babylonNode.scaling = scaling;
    }
    // eslint-disable-next-line @typescript-eslint/promise-function-async, no-restricted-syntax
    _loadSkinAsync(context, node, skin, assign) {
        if (!this._parent.loadSkins) {
            return Promise.resolve();
        }
        const extensionPromise = this._extensionsLoadSkinAsync(context, node, skin);
        if (extensionPromise) {
            return extensionPromise;
        }
        if (skin._data) {
            assign(skin._data.babylonSkeleton);
            return skin._data.promise;
        }
        const skeletonId = `skeleton${skin.index}`;
        this._babylonScene._blockEntityCollection = !!this._assetContainer;
        const babylonSkeleton = new Skeleton(skin.name || skeletonId, skeletonId, this._babylonScene);
        babylonSkeleton._parentContainer = this._assetContainer;
        this._babylonScene._blockEntityCollection = false;
        this._loadBones(context, skin, babylonSkeleton);
        const promise = this._loadSkinInverseBindMatricesDataAsync(context, skin).then((inverseBindMatricesData) => {
            this._updateBoneMatrices(babylonSkeleton, inverseBindMatricesData);
        });
        skin._data = {
            babylonSkeleton: babylonSkeleton,
            promise: promise,
        };
        assign(babylonSkeleton);
        return promise;
    }
    _loadBones(context, skin, babylonSkeleton) {
        if (skin.skeleton == undefined || this._parent.alwaysComputeSkeletonRootNode) {
            const rootNode = this._findSkeletonRootNode(`${context}/joints`, skin.joints);
            if (rootNode) {
                if (skin.skeleton === undefined) {
                    skin.skeleton = rootNode.index;
                }
                else {
                    const isParent = (a, b) => {
                        for (; b.parent; b = b.parent) {
                            if (b.parent === a) {
                                return true;
                            }
                        }
                        return false;
                    };
                    const skeletonNode = ArrayItem.Get(`${context}/skeleton`, this._gltf.nodes, skin.skeleton);
                    if (skeletonNode !== rootNode && !isParent(skeletonNode, rootNode)) {
                        Logger.Warn(`${context}/skeleton: Overriding with nearest common ancestor as skeleton node is not a common root`);
                        skin.skeleton = rootNode.index;
                    }
                }
            }
            else {
                Logger.Warn(`${context}: Failed to find common root`);
            }
        }
        const babylonBones = {};
        for (const index of skin.joints) {
            const node = ArrayItem.Get(`${context}/joints/${index}`, this._gltf.nodes, index);
            this._loadBone(node, skin, babylonSkeleton, babylonBones);
        }
    }
    _findSkeletonRootNode(context, joints) {
        if (joints.length === 0) {
            return null;
        }
        const paths = {};
        for (const index of joints) {
            const path = [];
            let node = ArrayItem.Get(`${context}/${index}`, this._gltf.nodes, index);
            while (node.index !== -1) {
                path.unshift(node);
                node = node.parent;
            }
            paths[index] = path;
        }
        let rootNode = null;
        for (let i = 0;; ++i) {
            let path = paths[joints[0]];
            if (i >= path.length) {
                return rootNode;
            }
            const node = path[i];
            for (let j = 1; j < joints.length; ++j) {
                path = paths[joints[j]];
                if (i >= path.length || node !== path[i]) {
                    return rootNode;
                }
            }
            rootNode = node;
        }
    }
    _loadBone(node, skin, babylonSkeleton, babylonBones) {
        node._isJoint = true;
        let babylonBone = babylonBones[node.index];
        if (babylonBone) {
            return babylonBone;
        }
        let parentBabylonBone = null;
        if (node.index !== skin.skeleton) {
            if (node.parent && node.parent.index !== -1) {
                parentBabylonBone = this._loadBone(node.parent, skin, babylonSkeleton, babylonBones);
            }
            else if (skin.skeleton !== undefined) {
                Logger.Warn(`/skins/${skin.index}/skeleton: Skeleton node is not a common root`);
            }
        }
        const boneIndex = skin.joints.indexOf(node.index);
        babylonBone = new Bone(node.name || `joint${node.index}`, babylonSkeleton, parentBabylonBone, this._getNodeMatrix(node), null, null, boneIndex);
        babylonBones[node.index] = babylonBone;
        // Wait until the scene is loaded to ensure the transform nodes are loaded.
        this._postSceneLoadActions.push(() => {
            // Link the Babylon bone with the corresponding Babylon transform node.
            // A glTF joint is a pointer to a glTF node in the glTF node hierarchy similar to Unity3D.
            babylonBone.linkTransformNode(node._babylonTransformNode);
        });
        return babylonBone;
    }
    _loadSkinInverseBindMatricesDataAsync(context, skin) {
        if (skin.inverseBindMatrices == undefined) {
            return Promise.resolve(null);
        }
        const accessor = ArrayItem.Get(`${context}/inverseBindMatrices`, this._gltf.accessors, skin.inverseBindMatrices);
        return this._loadFloatAccessorAsync(`/accessors/${accessor.index}`, accessor);
    }
    _updateBoneMatrices(babylonSkeleton, inverseBindMatricesData) {
        for (const babylonBone of babylonSkeleton.bones) {
            const baseMatrix = Matrix.Identity();
            const boneIndex = babylonBone._index;
            if (inverseBindMatricesData && boneIndex !== -1) {
                Matrix.FromArrayToRef(inverseBindMatricesData, boneIndex * 16, baseMatrix);
                baseMatrix.invertToRef(baseMatrix);
            }
            const babylonParentBone = babylonBone.getParent();
            if (babylonParentBone) {
                baseMatrix.multiplyToRef(babylonParentBone.getAbsoluteInverseBindMatrix(), baseMatrix);
            }
            babylonBone.updateMatrix(baseMatrix, false, false);
            babylonBone._updateAbsoluteBindMatrices(undefined, false);
        }
    }
    _getNodeMatrix(node) {
        return node.matrix
            ? Matrix.FromArray(node.matrix)
            : Matrix.Compose(node.scale ? Vector3.FromArray(node.scale) : Vector3.One(), node.rotation ? Quaternion.FromArray(node.rotation) : Quaternion.Identity(), node.translation ? Vector3.FromArray(node.translation) : Vector3.Zero());
    }
    /**
     * Loads a glTF camera.
     * @param context The context when loading the asset
     * @param camera The glTF camera property
     * @param assign A function called synchronously after parsing the glTF properties
     * @returns A promise that resolves with the loaded Babylon camera when the load is complete
     */
    loadCameraAsync(context, camera, assign = () => { }) {
        const extensionPromise = this._extensionsLoadCameraAsync(context, camera, assign);
        if (extensionPromise) {
            return extensionPromise;
        }
        const promises = new Array();
        this.logOpen(`${context} ${camera.name || ""}`);
        this._babylonScene._blockEntityCollection = !!this._assetContainer;
        const babylonCamera = new FreeCamera(camera.name || `camera${camera.index}`, Vector3.Zero(), this._babylonScene, false);
        babylonCamera._parentContainer = this._assetContainer;
        this._babylonScene._blockEntityCollection = false;
        camera._babylonCamera = babylonCamera;
        // glTF cameras look towards the local -Z axis.
        babylonCamera.setTarget(new Vector3(0, 0, -1));
        switch (camera.type) {
            case "perspective" /* CameraType.PERSPECTIVE */: {
                const perspective = camera.perspective;
                if (!perspective) {
                    throw new Error(`${context}: Camera perspective properties are missing`);
                }
                babylonCamera.fov = perspective.yfov;
                babylonCamera.minZ = perspective.znear;
                babylonCamera.maxZ = perspective.zfar || 0;
                break;
            }
            case "orthographic" /* CameraType.ORTHOGRAPHIC */: {
                if (!camera.orthographic) {
                    throw new Error(`${context}: Camera orthographic properties are missing`);
                }
                babylonCamera.mode = Camera.ORTHOGRAPHIC_CAMERA;
                babylonCamera.orthoLeft = -camera.orthographic.xmag;
                babylonCamera.orthoRight = camera.orthographic.xmag;
                babylonCamera.orthoBottom = -camera.orthographic.ymag;
                babylonCamera.orthoTop = camera.orthographic.ymag;
                babylonCamera.minZ = camera.orthographic.znear;
                babylonCamera.maxZ = camera.orthographic.zfar;
                break;
            }
            default: {
                throw new Error(`${context}: Invalid camera type (${camera.type})`);
            }
        }
        GLTFLoader.AddPointerMetadata(babylonCamera, context);
        this._parent.onCameraLoadedObservable.notifyObservers(babylonCamera);
        assign(babylonCamera);
        this.logClose();
        return Promise.all(promises).then(() => {
            return babylonCamera;
        });
    }
    _loadAnimationsAsync() {
        this._parent._startPerformanceCounter("Load animations");
        const animations = this._gltf.animations;
        if (!animations) {
            return Promise.resolve();
        }
        const promises = new Array();
        for (let index = 0; index < animations.length; index++) {
            const animation = animations[index];
            promises.push(this.loadAnimationAsync(`/animations/${animation.index}`, animation).then((animationGroup) => {
                // Delete the animation group if it ended up not having any animations in it.
                if (animationGroup.targetedAnimations.length === 0) {
                    animationGroup.dispose();
                }
            }));
        }
        return Promise.all(promises).then(() => {
            this._parent._endPerformanceCounter("Load animations");
        });
    }
    /**
     * Loads a glTF animation.
     * @param context The context when loading the asset
     * @param animation The glTF animation property
     * @returns A promise that resolves with the loaded Babylon animation group when the load is complete
     */
    loadAnimationAsync(context, animation) {
        this._parent._startPerformanceCounter("Load animation");
        const promise = this._extensionsLoadAnimationAsync(context, animation);
        if (promise) {
            return promise;
        }
        // eslint-disable-next-line @typescript-eslint/naming-convention
        return LazyAnimationGroupModulePromise.value.then(({ AnimationGroup }) => {
            this._babylonScene._blockEntityCollection = !!this._assetContainer;
            const babylonAnimationGroup = new AnimationGroup(animation.name || `animation${animation.index}`, this._babylonScene);
            babylonAnimationGroup._parentContainer = this._assetContainer;
            this._babylonScene._blockEntityCollection = false;
            animation._babylonAnimationGroup = babylonAnimationGroup;
            const promises = new Array();
            ArrayItem.Assign(animation.channels);
            ArrayItem.Assign(animation.samplers);
            for (const channel of animation.channels) {
                promises.push(this._loadAnimationChannelAsync(`${context}/channels/${channel.index}`, context, animation, channel, (babylonTarget, babylonAnimation) => {
                    babylonTarget.animations = babylonTarget.animations || [];
                    babylonTarget.animations.push(babylonAnimation);
                    babylonAnimationGroup.addTargetedAnimation(babylonAnimation, babylonTarget);
                }));
            }
            this._parent._endPerformanceCounter("Load animation");
            return Promise.all(promises).then(() => {
                babylonAnimationGroup.normalize(0);
                return babylonAnimationGroup;
            });
        });
    }
    /**
     * @hidden
     * Loads a glTF animation channel.
     * @param context The context when loading the asset
     * @param animationContext The context of the animation when loading the asset
     * @param animation The glTF animation property
     * @param channel The glTF animation channel property
     * @param onLoad Called for each animation loaded
     * @returns A void promise that resolves when the load is complete
     */
    _loadAnimationChannelAsync(context, animationContext, animation, channel, onLoad) {
        const promise = this._extensionsLoadAnimationChannelAsync(context, animationContext, animation, channel, onLoad);
        if (promise) {
            return promise;
        }
        if (channel.target.node == undefined) {
            return Promise.resolve();
        }
        const targetNode = ArrayItem.Get(`${context}/target/node`, this._gltf.nodes, channel.target.node);
        const channelTargetPath = channel.target.path;
        const pathIsWeights = channelTargetPath === "weights" /* AnimationChannelTargetPath.WEIGHTS */;
        // Ignore animations that have no animation targets.
        if ((pathIsWeights && !targetNode._numMorphTargets) || (!pathIsWeights && !targetNode._babylonTransformNode)) {
            return Promise.resolve();
        }
        // Don't load node animations if disabled.
        if (!this._parent.loadNodeAnimations && !pathIsWeights && !targetNode._isJoint) {
            return Promise.resolve();
        }
        // async-load the animation sampler to provide the interpolation of the channelTargetPath
        return LazyLoaderAnimationModulePromise.value.then(() => {
            let properties;
            switch (channelTargetPath) {
                case "translation" /* AnimationChannelTargetPath.TRANSLATION */: {
                    properties = GetMappingForKey("/nodes/{}/translation")?.interpolation;
                    break;
                }
                case "rotation" /* AnimationChannelTargetPath.ROTATION */: {
                    properties = GetMappingForKey("/nodes/{}/rotation")?.interpolation;
                    break;
                }
                case "scale" /* AnimationChannelTargetPath.SCALE */: {
                    properties = GetMappingForKey("/nodes/{}/scale")?.interpolation;
                    break;
                }
                case "weights" /* AnimationChannelTargetPath.WEIGHTS */: {
                    properties = GetMappingForKey("/nodes/{}/weights")?.interpolation;
                    break;
                }
                default: {
                    throw new Error(`${context}/target/path: Invalid value (${channel.target.path})`);
                }
            }
            // stay safe
            if (!properties) {
                throw new Error(`${context}/target/path: Could not find interpolation properties for target path (${channel.target.path})`);
            }
            const targetInfo = {
                object: targetNode,
                info: properties,
            };
            return this._loadAnimationChannelFromTargetInfoAsync(context, animationContext, animation, channel, targetInfo, onLoad);
        });
    }
    /**
     * @hidden
     * Loads a glTF animation channel.
     * @param context The context when loading the asset
     * @param animationContext The context of the animation when loading the asset
     * @param animation The glTF animation property
     * @param channel The glTF animation channel property
     * @param targetInfo The glTF target and properties
     * @param onLoad Called for each animation loaded
     * @returns A void promise that resolves when the load is complete
     */
    _loadAnimationChannelFromTargetInfoAsync(context, animationContext, animation, channel, targetInfo, onLoad) {
        const fps = this.parent.targetFps;
        const invfps = 1 / fps;
        const sampler = ArrayItem.Get(`${context}/sampler`, animation.samplers, channel.sampler);
        return this._loadAnimationSamplerAsync(`${animationContext}/samplers/${channel.sampler}`, sampler).then((data) => {
            let numAnimations = 0;
            const target = targetInfo.object;
            const propertyInfos = targetInfo.info;
            // Extract the corresponding values from the read value.
            // GLTF values may be dispatched to several Babylon properties.
            // For example, baseColorFactor [`r`, `g`, `b`, `a`] is dispatched to
            // - albedoColor as Color3(`r`, `g`, `b`)
            // - alpha as `a`
            for (const propertyInfo of propertyInfos) {
                const stride = propertyInfo.getStride(target);
                const input = data.input;
                const output = data.output;
                const keys = new Array(input.length);
                let outputOffset = 0;
                switch (data.interpolation) {
                    case "STEP" /* AnimationSamplerInterpolation.STEP */: {
                        for (let index = 0; index < input.length; index++) {
                            const value = propertyInfo.getValue(target, output, outputOffset, 1);
                            outputOffset += stride;
                            keys[index] = {
                                frame: input[index] * fps,
                                value: value,
                                interpolation: 1 /* AnimationKeyInterpolation.STEP */,
                            };
                        }
                        break;
                    }
                    case "CUBICSPLINE" /* AnimationSamplerInterpolation.CUBICSPLINE */: {
                        for (let index = 0; index < input.length; index++) {
                            const inTangent = propertyInfo.getValue(target, output, outputOffset, invfps);
                            outputOffset += stride;
                            const value = propertyInfo.getValue(target, output, outputOffset, 1);
                            outputOffset += stride;
                            const outTangent = propertyInfo.getValue(target, output, outputOffset, invfps);
                            outputOffset += stride;
                            keys[index] = {
                                frame: input[index] * fps,
                                inTangent: inTangent,
                                value: value,
                                outTangent: outTangent,
                            };
                        }
                        break;
                    }
                    case "LINEAR" /* AnimationSamplerInterpolation.LINEAR */: {
                        for (let index = 0; index < input.length; index++) {
                            const value = propertyInfo.getValue(target, output, outputOffset, 1);
                            outputOffset += stride;
                            keys[index] = {
                                frame: input[index] * fps,
                                value: value,
                            };
                        }
                        break;
                    }
                }
                if (outputOffset > 0) {
                    const name = `${animation.name || `animation${animation.index}`}_channel${channel.index}_${numAnimations}`;
                    const babylonAnimations = propertyInfo.buildAnimations(target, name, fps, keys);
                    for (const babylonAnimation of babylonAnimations) {
                        numAnimations++;
                        onLoad(babylonAnimation.babylonAnimatable, babylonAnimation.babylonAnimation);
                    }
                }
            }
        });
    }
    _loadAnimationSamplerAsync(context, sampler) {
        if (sampler._data) {
            return sampler._data;
        }
        const interpolation = sampler.interpolation || "LINEAR" /* AnimationSamplerInterpolation.LINEAR */;
        switch (interpolation) {
            case "STEP" /* AnimationSamplerInterpolation.STEP */:
            case "LINEAR" /* AnimationSamplerInterpolation.LINEAR */:
            case "CUBICSPLINE" /* AnimationSamplerInterpolation.CUBICSPLINE */: {
                break;
            }
            default: {
                throw new Error(`${context}/interpolation: Invalid value (${sampler.interpolation})`);
            }
        }
        const inputAccessor = ArrayItem.Get(`${context}/input`, this._gltf.accessors, sampler.input);
        const outputAccessor = ArrayItem.Get(`${context}/output`, this._gltf.accessors, sampler.output);
        sampler._data = Promise.all([
            this._loadFloatAccessorAsync(`/accessors/${inputAccessor.index}`, inputAccessor),
            this._loadFloatAccessorAsync(`/accessors/${outputAccessor.index}`, outputAccessor),
        ]).then(([inputData, outputData]) => {
            return {
                input: inputData,
                interpolation: interpolation,
                output: outputData,
            };
        });
        return sampler._data;
    }
    /**
     * Loads a glTF buffer.
     * @param context The context when loading the asset
     * @param buffer The glTF buffer property
     * @param byteOffset The byte offset to use
     * @param byteLength The byte length to use
     * @returns A promise that resolves with the loaded data when the load is complete
     */
    loadBufferAsync(context, buffer, byteOffset, byteLength) {
        const extensionPromise = this._extensionsLoadBufferAsync(context, buffer, byteOffset, byteLength);
        if (extensionPromise) {
            return extensionPromise;
        }
        if (!buffer._data) {
            if (buffer.uri) {
                buffer._data = this.loadUriAsync(`${context}/uri`, buffer, buffer.uri);
            }
            else {
                if (!this._bin) {
                    throw new Error(`${context}: Uri is missing or the binary glTF is missing its binary chunk`);
                }
                buffer._data = this._bin.readAsync(0, buffer.byteLength);
            }
        }
        return buffer._data.then((data) => {
            try {
                return new Uint8Array(data.buffer, data.byteOffset + byteOffset, byteLength);
            }
            catch (e) {
                throw new Error(`${context}: ${e.message}`);
            }
        });
    }
    /**
     * Loads a glTF buffer view.
     * @param context The context when loading the asset
     * @param bufferView The glTF buffer view property
     * @returns A promise that resolves with the loaded data when the load is complete
     */
    loadBufferViewAsync(context, bufferView) {
        const extensionPromise = this._extensionsLoadBufferViewAsync(context, bufferView);
        if (extensionPromise) {
            return extensionPromise;
        }
        if (bufferView._data) {
            return bufferView._data;
        }
        const buffer = ArrayItem.Get(`${context}/buffer`, this._gltf.buffers, bufferView.buffer);
        bufferView._data = this.loadBufferAsync(`/buffers/${buffer.index}`, buffer, bufferView.byteOffset || 0, bufferView.byteLength);
        return bufferView._data;
    }
    _loadAccessorAsync(context, accessor, constructor) {
        if (accessor._data) {
            return accessor._data;
        }
        const numComponents = GLTFLoader._GetNumComponents(context, accessor.type);
        const byteStride = numComponents * VertexBuffer.GetTypeByteLength(accessor.componentType);
        const length = numComponents * accessor.count;
        if (accessor.bufferView == undefined) {
            accessor._data = Promise.resolve(new constructor(length));
        }
        else {
            const bufferView = ArrayItem.Get(`${context}/bufferView`, this._gltf.bufferViews, accessor.bufferView);
            accessor._data = this.loadBufferViewAsync(`/bufferViews/${bufferView.index}`, bufferView).then((data) => {
                if (accessor.componentType === 5126 /* AccessorComponentType.FLOAT */ && !accessor.normalized && (!bufferView.byteStride || bufferView.byteStride === byteStride)) {
                    return GLTFLoader._GetTypedArray(context, accessor.componentType, data, accessor.byteOffset, length);
                }
                else {
                    const typedArray = new constructor(length);
                    VertexBuffer.ForEach(data, accessor.byteOffset || 0, bufferView.byteStride || byteStride, numComponents, accessor.componentType, typedArray.length, accessor.normalized || false, (value, index) => {
                        typedArray[index] = value;
                    });
                    return typedArray;
                }
            });
        }
        if (accessor.sparse) {
            const sparse = accessor.sparse;
            accessor._data = accessor._data.then((data) => {
                const typedArray = data;
                const indicesBufferView = ArrayItem.Get(`${context}/sparse/indices/bufferView`, this._gltf.bufferViews, sparse.indices.bufferView);
                const valuesBufferView = ArrayItem.Get(`${context}/sparse/values/bufferView`, this._gltf.bufferViews, sparse.values.bufferView);
                return Promise.all([
                    this.loadBufferViewAsync(`/bufferViews/${indicesBufferView.index}`, indicesBufferView),
                    this.loadBufferViewAsync(`/bufferViews/${valuesBufferView.index}`, valuesBufferView),
                ]).then(([indicesData, valuesData]) => {
                    const indices = GLTFLoader._GetTypedArray(`${context}/sparse/indices`, sparse.indices.componentType, indicesData, sparse.indices.byteOffset, sparse.count);
                    const sparseLength = numComponents * sparse.count;
                    let values;
                    if (accessor.componentType === 5126 /* AccessorComponentType.FLOAT */ && !accessor.normalized) {
                        values = GLTFLoader._GetTypedArray(`${context}/sparse/values`, accessor.componentType, valuesData, sparse.values.byteOffset, sparseLength);
                    }
                    else {
                        const sparseData = GLTFLoader._GetTypedArray(`${context}/sparse/values`, accessor.componentType, valuesData, sparse.values.byteOffset, sparseLength);
                        values = new constructor(sparseLength);
                        VertexBuffer.ForEach(sparseData, 0, byteStride, numComponents, accessor.componentType, values.length, accessor.normalized || false, (value, index) => {
                            values[index] = value;
                        });
                    }
                    let valuesIndex = 0;
                    for (let indicesIndex = 0; indicesIndex < indices.length; indicesIndex++) {
                        let dataIndex = indices[indicesIndex] * numComponents;
                        for (let componentIndex = 0; componentIndex < numComponents; componentIndex++) {
                            typedArray[dataIndex++] = values[valuesIndex++];
                        }
                    }
                    return typedArray;
                });
            });
        }
        return accessor._data;
    }
    /**
     * @internal
     */
    _loadFloatAccessorAsync(context, accessor) {
        return this._loadAccessorAsync(context, accessor, Float32Array);
    }
    /**
     * @internal
     */
    _loadIndicesAccessorAsync(context, accessor) {
        if (accessor.type !== "SCALAR" /* AccessorType.SCALAR */) {
            throw new Error(`${context}/type: Invalid value ${accessor.type}`);
        }
        if (accessor.componentType !== 5121 /* AccessorComponentType.UNSIGNED_BYTE */ &&
            accessor.componentType !== 5123 /* AccessorComponentType.UNSIGNED_SHORT */ &&
            accessor.componentType !== 5125 /* AccessorComponentType.UNSIGNED_INT */) {
            throw new Error(`${context}/componentType: Invalid value ${accessor.componentType}`);
        }
        if (accessor._data) {
            return accessor._data;
        }
        if (accessor.sparse) {
            const constructor = GLTFLoader._GetTypedArrayConstructor(`${context}/componentType`, accessor.componentType);
            accessor._data = this._loadAccessorAsync(context, accessor, constructor);
        }
        else {
            const bufferView = ArrayItem.Get(`${context}/bufferView`, this._gltf.bufferViews, accessor.bufferView);
            accessor._data = this.loadBufferViewAsync(`/bufferViews/${bufferView.index}`, bufferView).then((data) => {
                return GLTFLoader._GetTypedArray(context, accessor.componentType, data, accessor.byteOffset, accessor.count);
            });
        }
        return accessor._data;
    }
    /**
     * @internal
     */
    _loadVertexBufferViewAsync(bufferView) {
        if (bufferView._babylonBuffer) {
            return bufferView._babylonBuffer;
        }
        const engine = this._babylonScene.getEngine();
        bufferView._babylonBuffer = this.loadBufferViewAsync(`/bufferViews/${bufferView.index}`, bufferView).then((data) => {
            return new Buffer(engine, data, false);
        });
        return bufferView._babylonBuffer;
    }
    /**
     * @internal
     */
    _loadVertexAccessorAsync(context, accessor, kind) {
        if (accessor._babylonVertexBuffer?.[kind]) {
            return accessor._babylonVertexBuffer[kind];
        }
        if (!accessor._babylonVertexBuffer) {
            accessor._babylonVertexBuffer = {};
        }
        const engine = this._babylonScene.getEngine();
        if (accessor.sparse || accessor.bufferView == undefined) {
            accessor._babylonVertexBuffer[kind] = this._loadFloatAccessorAsync(context, accessor).then((data) => {
                return new VertexBuffer(engine, data, kind, false);
            });
        }
        else {
            const bufferView = ArrayItem.Get(`${context}/bufferView`, this._gltf.bufferViews, accessor.bufferView);
            accessor._babylonVertexBuffer[kind] = this._loadVertexBufferViewAsync(bufferView).then((babylonBuffer) => {
                const numComponents = GLTFLoader._GetNumComponents(context, accessor.type);
                return new VertexBuffer(engine, babylonBuffer, kind, false, undefined, bufferView.byteStride, undefined, accessor.byteOffset, numComponents, accessor.componentType, accessor.normalized, true, undefined, true);
            });
        }
        return accessor._babylonVertexBuffer[kind];
    }
    _loadMaterialMetallicRoughnessPropertiesAsync(context, properties, babylonMaterial) {
        const promises = new Array();
        const adapter = this._getOrCreateMaterialAdapter(babylonMaterial);
        if (properties) {
            // Set base color and alpha using adapter
            if (properties.baseColorFactor) {
                adapter.baseColor = Color3.FromArray(properties.baseColorFactor);
                adapter.geometryOpacity = properties.baseColorFactor[3];
            }
            else {
                adapter.baseColor = Color3.White();
            }
            // Set metallic and roughness using adapter
            adapter.baseMetalness = properties.metallicFactor == undefined ? 1 : properties.metallicFactor;
            adapter.specularRoughness = properties.roughnessFactor == undefined ? 1 : properties.roughnessFactor;
            if (properties.baseColorTexture) {
                promises.push(this.loadTextureInfoAsync(`${context}/baseColorTexture`, properties.baseColorTexture, (texture) => {
                    texture.name = `${babylonMaterial.name} (Base Color)`;
                    adapter.baseColorTexture = texture;
                }));
            }
            if (properties.metallicRoughnessTexture) {
                properties.metallicRoughnessTexture.nonColorData = true;
                promises.push(this.loadTextureInfoAsync(`${context}/metallicRoughnessTexture`, properties.metallicRoughnessTexture, (texture) => {
                    texture.name = `${babylonMaterial.name} (Metallic Roughness)`;
                    adapter.baseMetalnessTexture = texture;
                    adapter.specularRoughnessTexture = texture;
                }));
                // Configure texture channel usage using adapter
                adapter.useRoughnessFromMetallicTextureGreen = true;
                adapter.useMetallicFromMetallicTextureBlue = true;
            }
        }
        return Promise.all(promises).then(() => { });
    }
    /**
     * @internal
     */
    _loadMaterialAsync(context, material, babylonMesh, babylonDrawMode, assign = () => { }) {
        const extensionPromise = this._extensionsLoadMaterialAsync(context, material, babylonMesh, babylonDrawMode, assign);
        if (extensionPromise) {
            return extensionPromise;
        }
        material._data = material._data || {};
        let babylonData = material._data[babylonDrawMode];
        if (!babylonData) {
            this.logOpen(`${context} ${material.name || ""}`);
            const babylonMaterial = this.createMaterial(context, material, babylonDrawMode);
            babylonData = {
                babylonMaterial: babylonMaterial,
                babylonMeshes: [],
                promise: this.loadMaterialPropertiesAsync(context, material, babylonMaterial),
            };
            material._data[babylonDrawMode] = babylonData;
            GLTFLoader.AddPointerMetadata(babylonMaterial, context);
            this._parent.onMaterialLoadedObservable.notifyObservers(babylonMaterial);
            this.logClose();
        }
        if (babylonMesh) {
            babylonData.babylonMeshes.push(babylonMesh);
            babylonMesh.onDisposeObservable.addOnce(() => {
                const index = babylonData.babylonMeshes.indexOf(babylonMesh);
                if (index !== -1) {
                    babylonData.babylonMeshes.splice(index, 1);
                }
            });
        }
        assign(babylonData.babylonMaterial);
        return babylonData.promise.then(() => {
            return babylonData.babylonMaterial;
        });
    }
    _createDefaultMaterial(name, babylonDrawMode) {
        if (!this._pbrMaterialImpl) {
            throw new Error("PBR Material class not loaded");
        }
        this._babylonScene._blockEntityCollection = !!this._assetContainer;
        const babylonMaterial = new this._pbrMaterialImpl.materialClass(name, this._babylonScene);
        babylonMaterial._parentContainer = this._assetContainer;
        this._babylonScene._blockEntityCollection = false;
        babylonMaterial.fillMode = babylonDrawMode;
        babylonMaterial.transparencyMode = this._pbrMaterialImpl.materialClass.MATERIAL_OPAQUE;
        // Create the material adapter and set some default properties.
        // We don't need to wait for the promise to resolve here.
        const adapter = this._getOrCreateMaterialAdapter(babylonMaterial);
        adapter.transparencyAsAlphaCoverage = this._parent.transparencyAsCoverage;
        // Set default metallic and roughness values
        adapter.baseMetalness = 1.0;
        adapter.specularRoughness = 1.0;
        return babylonMaterial;
    }
    /**
     * Creates a Babylon material from a glTF material.
     * @param context The context when loading the asset
     * @param material The glTF material property
     * @param babylonDrawMode The draw mode for the Babylon material
     * @returns The Babylon material
     */
    createMaterial(context, material, babylonDrawMode) {
        const extensionMaterial = this._extensionsCreateMaterial(context, material, babylonDrawMode);
        if (extensionMaterial) {
            return extensionMaterial;
        }
        const name = material.name || `material${material.index}`;
        const babylonMaterial = this._createDefaultMaterial(name, babylonDrawMode);
        return babylonMaterial;
    }
    /**
     * Loads properties from a glTF material into a Babylon material.
     * @param context The context when loading the asset
     * @param material The glTF material property
     * @param babylonMaterial The Babylon material
     * @returns A promise that resolves when the load is complete
     */
    loadMaterialPropertiesAsync(context, material, babylonMaterial) {
        const extensionPromise = this._extensionsLoadMaterialPropertiesAsync(context, material, babylonMaterial);
        if (extensionPromise) {
            return extensionPromise;
        }
        const promises = new Array();
        promises.push(this.loadMaterialBasePropertiesAsync(context, material, babylonMaterial));
        if (material.pbrMetallicRoughness) {
            promises.push(this._loadMaterialMetallicRoughnessPropertiesAsync(`${context}/pbrMetallicRoughness`, material.pbrMetallicRoughness, babylonMaterial));
        }
        this.loadMaterialAlphaProperties(context, material, babylonMaterial);
        return Promise.all(promises).then(() => { });
    }
    /**
     * Loads the normal, occlusion, and emissive properties from a glTF material into a Babylon material.
     * @param context The context when loading the asset
     * @param material The glTF material property
     * @param babylonMaterial The Babylon material
     * @returns A promise that resolves when the load is complete
     */
    loadMaterialBasePropertiesAsync(context, material, babylonMaterial) {
        const promises = new Array();
        const adapter = this._getOrCreateMaterialAdapter(babylonMaterial);
        // Set emission color using adapter
        adapter.emissionColor = material.emissiveFactor ? Color3.FromArray(material.emissiveFactor) : new Color3(0, 0, 0);
        // Set double-sided properties using adapter
        if (material.doubleSided) {
            adapter.backFaceCulling = false;
            adapter.twoSidedLighting = true;
        }
        if (material.normalTexture) {
            material.normalTexture.nonColorData = true;
            promises.push(this.loadTextureInfoAsync(`${context}/normalTexture`, material.normalTexture, (texture) => {
                texture.name = `${babylonMaterial.name} (Normal)`;
                adapter.geometryNormalTexture = texture;
                if (material.normalTexture?.scale != undefined) {
                    texture.level = material.normalTexture.scale;
                }
            }));
            // Set normal map inversions using adapter
            adapter.setNormalMapInversions(!this._babylonScene.useRightHandedSystem, this._babylonScene.useRightHandedSystem);
        }
        let aoTexture;
        let aoStrength = 1.0;
        let emissionTexture;
        if (material.occlusionTexture) {
            material.occlusionTexture.nonColorData = true;
            promises.push(this.loadTextureInfoAsync(`${context}/occlusionTexture`, material.occlusionTexture, (texture) => {
                texture.name = `${babylonMaterial.name} (Occlusion)`;
                aoTexture = texture;
            }));
            if (material.occlusionTexture.strength != undefined) {
                aoStrength = material.occlusionTexture.strength;
            }
        }
        if (material.emissiveTexture) {
            promises.push(this.loadTextureInfoAsync(`${context}/emissiveTexture`, material.emissiveTexture, (texture) => {
                texture.name = `${babylonMaterial.name} (Emissive)`;
                emissionTexture = texture;
            }));
        }
        return Promise.all(promises).then(() => {
            // Set ambient occlusion and emissive textures using adapter
            if (aoTexture) {
                adapter.ambientOcclusionTexture = aoTexture;
                adapter.ambientOcclusionTextureStrength = aoStrength;
            }
            if (emissionTexture) {
                adapter.emissionColorTexture = emissionTexture;
            }
        });
    }
    /**
     * Loads the alpha properties from a glTF material into a Babylon material.
     * Must be called after the setting the albedo texture of the Babylon material when the material has an albedo texture.
     * @param context The context when loading the asset
     * @param material The glTF material property
     * @param babylonMaterial The Babylon material
     */
    loadMaterialAlphaProperties(context, material, babylonMaterial) {
        if (!this._pbrMaterialImpl) {
            throw new Error(`${context}: Material type not supported`);
        }
        const adapter = this._getOrCreateMaterialAdapter(babylonMaterial);
        const baseColorTexture = adapter.baseColorTexture;
        const alphaMode = material.alphaMode || "OPAQUE" /* MaterialAlphaMode.OPAQUE */;
        switch (alphaMode) {
            case "OPAQUE" /* MaterialAlphaMode.OPAQUE */: {
                babylonMaterial.transparencyMode = this._pbrMaterialImpl.materialClass.MATERIAL_OPAQUE;
                babylonMaterial.alpha = 1.0; // Force alpha to 1.0 for opaque mode.
                break;
            }
            case "MASK" /* MaterialAlphaMode.MASK */: {
                babylonMaterial.transparencyMode = this._pbrMaterialImpl.materialClass.MATERIAL_ALPHATEST;
                adapter.alphaCutOff = material.alphaCutoff == undefined ? 0.5 : material.alphaCutoff;
                if (baseColorTexture) {
                    baseColorTexture.hasAlpha = true;
                }
                break;
            }
            case "BLEND" /* MaterialAlphaMode.BLEND */: {
                babylonMaterial.transparencyMode = this._pbrMaterialImpl.materialClass.MATERIAL_ALPHABLEND;
                if (baseColorTexture) {
                    baseColorTexture.hasAlpha = true;
                    adapter.useAlphaFromBaseColorTexture = true;
                }
                break;
            }
            default: {
                throw new Error(`${context}/alphaMode: Invalid value (${material.alphaMode})`);
            }
        }
    }
    /**
     * Loads a glTF texture info.
     * @param context The context when loading the asset
     * @param textureInfo The glTF texture info property
     * @param assign A function called synchronously after parsing the glTF properties
     * @returns A promise that resolves with the loaded Babylon texture when the load is complete
     */
    loadTextureInfoAsync(context, textureInfo, assign = () => { }) {
        const extensionPromise = this._extensionsLoadTextureInfoAsync(context, textureInfo, assign);
        if (extensionPromise) {
            return extensionPromise;
        }
        this.logOpen(`${context}`);
        if (textureInfo.texCoord >= 6) {
            throw new Error(`${context}/texCoord: Invalid value (${textureInfo.texCoord})`);
        }
        const texture = ArrayItem.Get(`${context}/index`, this._gltf.textures, textureInfo.index);
        texture._textureInfo = textureInfo;
        const promise = this._loadTextureAsync(`/textures/${textureInfo.index}`, texture, (babylonTexture) => {
            babylonTexture.coordinatesIndex = textureInfo.texCoord || 0;
            GLTFLoader.AddPointerMetadata(babylonTexture, context);
            this._parent.onTextureLoadedObservable.notifyObservers(babylonTexture);
            assign(babylonTexture);
        });
        this.logClose();
        return promise;
    }
    /**
     * @internal
     */
    _loadTextureAsync(context, texture, assign = () => { }) {
        const extensionPromise = this._extensionsLoadTextureAsync(context, texture, assign);
        if (extensionPromise) {
            return extensionPromise;
        }
        this.logOpen(`${context} ${texture.name || ""}`);
        const sampler = texture.sampler == undefined ? GLTFLoader.DefaultSampler : ArrayItem.Get(`${context}/sampler`, this._gltf.samplers, texture.sampler);
        const image = ArrayItem.Get(`${context}/source`, this._gltf.images, texture.source);
        const promise = this._createTextureAsync(context, sampler, image, assign, undefined, !texture._textureInfo.nonColorData);
        this.logClose();
        return promise;
    }
    /**
     * @internal
     */
    _createTextureAsync(context, sampler, image, assign = () => { }, textureLoaderOptions, useSRGBBuffer) {
        const samplerData = this._loadSampler(`/samplers/${sampler.index}`, sampler);
        const promises = new Array();
        const deferred = new Deferred();
        this._babylonScene._blockEntityCollection = !!this._assetContainer;
        const textureCreationOptions = {
            noMipmap: samplerData.noMipMaps,
            invertY: false,
            samplingMode: samplerData.samplingMode,
            onLoad: () => {
                if (!this._disposed) {
                    deferred.resolve();
                }
            },
            onError: (message, exception) => {
                if (!this._disposed) {
                    deferred.reject(new Error(`${context}: ${exception && exception.message ? exception.message : message || "Failed to load texture"}`));
                }
            },
            mimeType: image.mimeType ?? GetMimeType(image.uri ?? ""),
            loaderOptions: textureLoaderOptions,
            useSRGBBuffer: !!useSRGBBuffer && this._parent.useSRGBBuffers,
        };
        const babylonTexture = new Texture(null, this._babylonScene, textureCreationOptions);
        babylonTexture._parentContainer = this._assetContainer;
        this._babylonScene._blockEntityCollection = false;
        promises.push(deferred.promise);
        promises.push(this.loadImageAsync(`/images/${image.index}`, image).then((data) => {
            const name = image.uri || `${this._fileName}#image${image.index}`;
            const dataUrl = `data:${this._uniqueRootUrl}${name}`;
            babylonTexture.updateURL(dataUrl, data);
            // Set the internal texture label.
            const internalTexture = babylonTexture.getInternalTexture();
            if (internalTexture) {
                internalTexture.label = image.name;
            }
        }));
        babylonTexture.wrapU = samplerData.wrapU;
        babylonTexture.wrapV = samplerData.wrapV;
        assign(babylonTexture);
        if (this._parent.useGltfTextureNames) {
            babylonTexture.name = image.name || image.uri || `image${image.index}`;
        }
        return Promise.all(promises).then(() => {
            return babylonTexture;
        });
    }
    _loadSampler(context, sampler) {
        if (!sampler._data) {
            sampler._data = {
                noMipMaps: sampler.minFilter === 9728 /* TextureMinFilter.NEAREST */ || sampler.minFilter === 9729 /* TextureMinFilter.LINEAR */,
                samplingMode: GLTFLoader._GetTextureSamplingMode(context, sampler),
                wrapU: GLTFLoader._GetTextureWrapMode(`${context}/wrapS`, sampler.wrapS),
                wrapV: GLTFLoader._GetTextureWrapMode(`${context}/wrapT`, sampler.wrapT),
            };
        }
        return sampler._data;
    }
    /**
     * Loads a glTF image.
     * @param context The context when loading the asset
     * @param image The glTF image property
     * @returns A promise that resolves with the loaded data when the load is complete
     */
    loadImageAsync(context, image) {
        if (!image._data) {
            this.logOpen(`${context} ${image.name || ""}`);
            if (image.uri) {
                image._data = this.loadUriAsync(`${context}/uri`, image, image.uri);
            }
            else {
                const bufferView = ArrayItem.Get(`${context}/bufferView`, this._gltf.bufferViews, image.bufferView);
                image._data = this.loadBufferViewAsync(`/bufferViews/${bufferView.index}`, bufferView);
            }
            this.logClose();
        }
        return image._data;
    }
    /**
     * Loads a glTF uri.
     * @param context The context when loading the asset
     * @param property The glTF property associated with the uri
     * @param uri The base64 or relative uri
     * @returns A promise that resolves with the loaded data when the load is complete
     */
    loadUriAsync(context, property, uri) {
        const extensionPromise = this._extensionsLoadUriAsync(context, property, uri);
        if (extensionPromise) {
            return extensionPromise;
        }
        if (!GLTFLoader._ValidateUri(uri)) {
            throw new Error(`${context}: '${uri}' is invalid`);
        }
        if (IsBase64DataUrl(uri)) {
            const data = new Uint8Array(DecodeBase64UrlToBinary(uri));
            this.log(`${context}: Decoded ${uri.substring(0, 64)}... (${data.length} bytes)`);
            return Promise.resolve(data);
        }
        this.log(`${context}: Loading ${uri}`);
        return this._parent.preprocessUrlAsync(this._rootUrl + uri).then((url) => {
            return new Promise((resolve, reject) => {
                this._parent._loadFile(this._babylonScene, url, (data) => {
                    if (!this._disposed) {
                        this.log(`${context}: Loaded ${uri} (${data.byteLength} bytes)`);
                        resolve(new Uint8Array(data));
                    }
                }, true, (request) => {
                    reject(new LoadFileError(`${context}: Failed to load '${uri}'${request ? ": " + request.status + " " + request.statusText : ""}`, request));
                });
            });
        });
    }
    /**
     * Adds a JSON pointer to the _internalMetadata of the Babylon object at `<object>._internalMetadata.gltf.pointers`.
     * @param babylonObject the Babylon object with _internalMetadata
     * @param pointer the JSON pointer
     */
    static AddPointerMetadata(babylonObject, pointer) {
        babylonObject.metadata = babylonObject.metadata || {};
        const metadata = (babylonObject._internalMetadata = babylonObject._internalMetadata || {});
        const gltf = (metadata.gltf = metadata.gltf || {});
        const pointers = (gltf.pointers = gltf.pointers || []);
        pointers.push(pointer);
    }
    static _GetTextureWrapMode(context, mode) {
        // Set defaults if undefined
        mode = mode == undefined ? 10497 /* TextureWrapMode.REPEAT */ : mode;
        switch (mode) {
            case 33071 /* TextureWrapMode.CLAMP_TO_EDGE */:
                return Texture.CLAMP_ADDRESSMODE;
            case 33648 /* TextureWrapMode.MIRRORED_REPEAT */:
                return Texture.MIRROR_ADDRESSMODE;
            case 10497 /* TextureWrapMode.REPEAT */:
                return Texture.WRAP_ADDRESSMODE;
            default:
                Logger.Warn(`${context}: Invalid value (${mode})`);
                return Texture.WRAP_ADDRESSMODE;
        }
    }
    static _GetTextureSamplingMode(context, sampler) {
        // Set defaults if undefined
        const magFilter = sampler.magFilter == undefined ? 9729 /* TextureMagFilter.LINEAR */ : sampler.magFilter;
        const minFilter = sampler.minFilter == undefined ? 9987 /* TextureMinFilter.LINEAR_MIPMAP_LINEAR */ : sampler.minFilter;
        if (magFilter === 9729 /* TextureMagFilter.LINEAR */) {
            switch (minFilter) {
                case 9728 /* TextureMinFilter.NEAREST */:
                    return Texture.LINEAR_NEAREST;
                case 9729 /* TextureMinFilter.LINEAR */:
                    return Texture.LINEAR_LINEAR;
                case 9984 /* TextureMinFilter.NEAREST_MIPMAP_NEAREST */:
                    return Texture.LINEAR_NEAREST_MIPNEAREST;
                case 9985 /* TextureMinFilter.LINEAR_MIPMAP_NEAREST */:
                    return Texture.LINEAR_LINEAR_MIPNEAREST;
                case 9986 /* TextureMinFilter.NEAREST_MIPMAP_LINEAR */:
                    return Texture.LINEAR_NEAREST_MIPLINEAR;
                case 9987 /* TextureMinFilter.LINEAR_MIPMAP_LINEAR */:
                    return Texture.LINEAR_LINEAR_MIPLINEAR;
                default:
                    Logger.Warn(`${context}/minFilter: Invalid value (${minFilter})`);
                    return Texture.LINEAR_LINEAR_MIPLINEAR;
            }
        }
        else {
            if (magFilter !== 9728 /* TextureMagFilter.NEAREST */) {
                Logger.Warn(`${context}/magFilter: Invalid value (${magFilter})`);
            }
            switch (minFilter) {
                case 9728 /* TextureMinFilter.NEAREST */:
                    return Texture.NEAREST_NEAREST;
                case 9729 /* TextureMinFilter.LINEAR */:
                    return Texture.NEAREST_LINEAR;
                case 9984 /* TextureMinFilter.NEAREST_MIPMAP_NEAREST */:
                    return Texture.NEAREST_NEAREST_MIPNEAREST;
                case 9985 /* TextureMinFilter.LINEAR_MIPMAP_NEAREST */:
                    return Texture.NEAREST_LINEAR_MIPNEAREST;
                case 9986 /* TextureMinFilter.NEAREST_MIPMAP_LINEAR */:
                    return Texture.NEAREST_NEAREST_MIPLINEAR;
                case 9987 /* TextureMinFilter.LINEAR_MIPMAP_LINEAR */:
                    return Texture.NEAREST_LINEAR_MIPLINEAR;
                default:
                    Logger.Warn(`${context}/minFilter: Invalid value (${minFilter})`);
                    return Texture.NEAREST_NEAREST_MIPNEAREST;
            }
        }
    }
    static _GetTypedArrayConstructor(context, componentType) {
        try {
            return GetTypedArrayConstructor(componentType);
        }
        catch (e) {
            throw new Error(`${context}: ${e.message}`);
        }
    }
    static _GetTypedArray(context, componentType, bufferView, byteOffset, length) {
        const buffer = bufferView.buffer;
        byteOffset = bufferView.byteOffset + (byteOffset || 0);
        const constructor = GLTFLoader._GetTypedArrayConstructor(`${context}/componentType`, componentType);
        const componentTypeLength = VertexBuffer.GetTypeByteLength(componentType);
        if (byteOffset % componentTypeLength !== 0) {
            // HACK: Copy the buffer if byte offset is not a multiple of component type byte length.
            Logger.Warn(`${context}: Copying buffer as byte offset (${byteOffset}) is not a multiple of component type byte length (${componentTypeLength})`);
            return new constructor(buffer.slice(byteOffset, byteOffset + length * componentTypeLength), 0);
        }
        return new constructor(buffer, byteOffset, length);
    }
    static _GetNumComponents(context, type) {
        switch (type) {
            case "SCALAR":
                return 1;
            case "VEC2":
                return 2;
            case "VEC3":
                return 3;
            case "VEC4":
                return 4;
            case "MAT2":
                return 4;
            case "MAT3":
                return 9;
            case "MAT4":
                return 16;
        }
        throw new Error(`${context}: Invalid type (${type})`);
    }
    static _ValidateUri(uri) {
        return Tools.IsBase64(uri) || uri.indexOf("..") === -1;
    }
    /**
     * @internal
     */
    static _GetDrawMode(context, mode) {
        if (mode == undefined) {
            mode = 4 /* MeshPrimitiveMode.TRIANGLES */;
        }
        switch (mode) {
            case 0 /* MeshPrimitiveMode.POINTS */:
                return Material.PointListDrawMode;
            case 1 /* MeshPrimitiveMode.LINES */:
                return Material.LineListDrawMode;
            case 2 /* MeshPrimitiveMode.LINE_LOOP */:
                return Material.LineLoopDrawMode;
            case 3 /* MeshPrimitiveMode.LINE_STRIP */:
                return Material.LineStripDrawMode;
            case 4 /* MeshPrimitiveMode.TRIANGLES */:
                return Material.TriangleFillMode;
            case 5 /* MeshPrimitiveMode.TRIANGLE_STRIP */:
                return Material.TriangleStripDrawMode;
            case 6 /* MeshPrimitiveMode.TRIANGLE_FAN */:
                return Material.TriangleFanDrawMode;
        }
        throw new Error(`${context}: Invalid mesh primitive mode (${mode})`);
    }
    _compileMaterialsAsync() {
        this._parent._startPerformanceCounter("Compile materials");
        const promises = new Array();
        if (this._gltf.materials) {
            for (const material of this._gltf.materials) {
                if (material._data) {
                    for (const babylonDrawMode in material._data) {
                        const babylonData = material._data[babylonDrawMode];
                        for (const babylonMesh of babylonData.babylonMeshes) {
                            // Ensure nonUniformScaling is set if necessary.
                            babylonMesh.computeWorldMatrix(true);
                            const babylonMaterial = babylonData.babylonMaterial;
                            promises.push(babylonMaterial.forceCompilationAsync(babylonMesh));
                            promises.push(babylonMaterial.forceCompilationAsync(babylonMesh, { useInstances: true }));
                            if (this._parent.useClipPlane) {
                                promises.push(babylonMaterial.forceCompilationAsync(babylonMesh, { clipPlane: true }));
                                promises.push(babylonMaterial.forceCompilationAsync(babylonMesh, { clipPlane: true, useInstances: true }));
                            }
                        }
                    }
                }
            }
        }
        return Promise.all(promises).then(() => {
            this._parent._endPerformanceCounter("Compile materials");
        });
    }
    _compileShadowGeneratorsAsync() {
        this._parent._startPerformanceCounter("Compile shadow generators");
        const promises = new Array();
        const lights = this._babylonScene.lights;
        for (const light of lights) {
            const generator = light.getShadowGenerator();
            if (generator) {
                promises.push(generator.forceCompilationAsync());
            }
        }
        return Promise.all(promises).then(() => {
            this._parent._endPerformanceCounter("Compile shadow generators");
        });
    }
    _forEachExtensions(action) {
        for (const extension of this._extensions) {
            if (extension.enabled) {
                action(extension);
            }
        }
    }
    _applyExtensions(property, functionName, actionAsync) {
        for (const extension of this._extensions) {
            if (extension.enabled) {
                const id = `${extension.name}.${functionName}`;
                const loaderProperty = property;
                loaderProperty._activeLoaderExtensionFunctions = loaderProperty._activeLoaderExtensionFunctions || {};
                const activeLoaderExtensionFunctions = loaderProperty._activeLoaderExtensionFunctions;
                if (!activeLoaderExtensionFunctions[id]) {
                    activeLoaderExtensionFunctions[id] = true;
                    try {
                        const result = actionAsync(extension);
                        if (result) {
                            return result;
                        }
                    }
                    finally {
                        delete activeLoaderExtensionFunctions[id];
                    }
                }
            }
        }
        return null;
    }
    _extensionsOnLoading() {
        this._forEachExtensions((extension) => extension.onLoading && extension.onLoading());
    }
    _extensionsOnReady() {
        this._forEachExtensions((extension) => extension.onReady && extension.onReady());
    }
    // eslint-disable-next-line no-restricted-syntax
    _extensionsLoadSceneAsync(context, scene) {
        return this._applyExtensions(scene, "loadScene", (extension) => extension.loadSceneAsync && extension.loadSceneAsync(context, scene));
    }
    // eslint-disable-next-line no-restricted-syntax
    _extensionsLoadNodeAsync(context, node, assign) {
        return this._applyExtensions(node, "loadNode", (extension) => extension.loadNodeAsync && extension.loadNodeAsync(context, node, assign));
    }
    // eslint-disable-next-line no-restricted-syntax
    _extensionsLoadCameraAsync(context, camera, assign) {
        return this._applyExtensions(camera, "loadCamera", (extension) => extension.loadCameraAsync && extension.loadCameraAsync(context, camera, assign));
    }
    // eslint-disable-next-line no-restricted-syntax
    _extensionsLoadVertexDataAsync(context, primitive, babylonMesh) {
        return this._applyExtensions(primitive, "loadVertexData", (extension) => extension._loadVertexDataAsync && extension._loadVertexDataAsync(context, primitive, babylonMesh));
    }
    // eslint-disable-next-line no-restricted-syntax
    _extensionsLoadMeshPrimitiveAsync(context, name, node, mesh, primitive, assign) {
        return this._applyExtensions(primitive, "loadMeshPrimitive", (extension) => extension._loadMeshPrimitiveAsync && extension._loadMeshPrimitiveAsync(context, name, node, mesh, primitive, assign));
    }
    // eslint-disable-next-line no-restricted-syntax
    _extensionsLoadMaterialAsync(context, material, babylonMesh, babylonDrawMode, assign) {
        return this._applyExtensions(material, "loadMaterial", (extension) => extension._loadMaterialAsync && extension._loadMaterialAsync(context, material, babylonMesh, babylonDrawMode, assign));
    }
    _extensionsCreateMaterial(context, material, babylonDrawMode) {
        return this._applyExtensions(material, "createMaterial", (extension) => extension.createMaterial && extension.createMaterial(context, material, babylonDrawMode));
    }
    // eslint-disable-next-line no-restricted-syntax
    _extensionsLoadMaterialPropertiesAsync(context, material, babylonMaterial) {
        return this._applyExtensions(material, "loadMaterialProperties", (extension) => extension.loadMaterialPropertiesAsync && extension.loadMaterialPropertiesAsync(context, material, babylonMaterial));
    }
    // eslint-disable-next-line no-restricted-syntax
    _extensionsLoadTextureInfoAsync(context, textureInfo, assign) {
        return this._applyExtensions(textureInfo, "loadTextureInfo", (extension) => extension.loadTextureInfoAsync && extension.loadTextureInfoAsync(context, textureInfo, assign));
    }
    // eslint-disable-next-line no-restricted-syntax
    _extensionsLoadTextureAsync(context, texture, assign) {
        return this._applyExtensions(texture, "loadTexture", (extension) => extension._loadTextureAsync && extension._loadTextureAsync(context, texture, assign));
    }
    // eslint-disable-next-line no-restricted-syntax
    _extensionsLoadAnimationAsync(context, animation) {
        return this._applyExtensions(animation, "loadAnimation", (extension) => extension.loadAnimationAsync && extension.loadAnimationAsync(context, animation));
    }
    // eslint-disable-next-line no-restricted-syntax
    _extensionsLoadAnimationChannelAsync(context, animationContext, animation, channel, onLoad) {
        return this._applyExtensions(animation, "loadAnimationChannel", (extension) => extension._loadAnimationChannelAsync && extension._loadAnimationChannelAsync(context, animationContext, animation, channel, onLoad));
    }
    // eslint-disable-next-line no-restricted-syntax
    _extensionsLoadSkinAsync(context, node, skin) {
        return this._applyExtensions(skin, "loadSkin", (extension) => extension._loadSkinAsync && extension._loadSkinAsync(context, node, skin));
    }
    // eslint-disable-next-line no-restricted-syntax
    _extensionsLoadUriAsync(context, property, uri) {
        return this._applyExtensions(property, "loadUri", (extension) => extension._loadUriAsync && extension._loadUriAsync(context, property, uri));
    }
    // eslint-disable-next-line no-restricted-syntax
    _extensionsLoadBufferViewAsync(context, bufferView) {
        return this._applyExtensions(bufferView, "loadBufferView", (extension) => extension.loadBufferViewAsync && extension.loadBufferViewAsync(context, bufferView));
    }
    // eslint-disable-next-line no-restricted-syntax
    _extensionsLoadBufferAsync(context, buffer, byteOffset, byteLength) {
        return this._applyExtensions(buffer, "loadBuffer", (extension) => extension.loadBufferAsync && extension.loadBufferAsync(context, buffer, byteOffset, byteLength));
    }
    /**
     * Helper method called by a loader extension to load an glTF extension.
     * @param context The context when loading the asset
     * @param property The glTF property to load the extension from
     * @param extensionName The name of the extension to load
     * @param actionAsync The action to run
     * @returns The promise returned by actionAsync or null if the extension does not exist
     */
    // eslint-disable-next-line no-restricted-syntax, @typescript-eslint/naming-convention
    static LoadExtensionAsync(context, property, extensionName, actionAsync) {
        if (!property.extensions) {
            return null;
        }
        const extensions = property.extensions;
        const extension = extensions[extensionName];
        if (!extension) {
            return null;
        }
        return actionAsync(`${context}/extensions/${extensionName}`, extension);
    }
    /**
     * Helper method called by a loader extension to load a glTF extra.
     * @param context The context when loading the asset
     * @param property The glTF property to load the extra from
     * @param extensionName The name of the extension to load
     * @param actionAsync The action to run
     * @returns The promise returned by actionAsync or null if the extra does not exist
     */
    // eslint-disable-next-line no-restricted-syntax, @typescript-eslint/naming-convention
    static LoadExtraAsync(context, property, extensionName, actionAsync) {
        if (!property.extras) {
            return null;
        }
        const extras = property.extras;
        const extra = extras[extensionName];
        if (!extra) {
            return null;
        }
        return actionAsync(`${context}/extras/${extensionName}`, extra);
    }
    /**
     * Checks for presence of an extension.
     * @param name The name of the extension to check
     * @returns A boolean indicating the presence of the given extension name in `extensionsUsed`
     */
    isExtensionUsed(name) {
        return !!this._gltf.extensionsUsed && this._gltf.extensionsUsed.indexOf(name) !== -1;
    }
    /**
     * Increments the indentation level and logs a message.
     * @param message The message to log
     */
    logOpen(message) {
        this._parent._logOpen(message);
    }
    /**
     * Decrements the indentation level.
     */
    logClose() {
        this._parent._logClose();
    }
    /**
     * Logs a message
     * @param message The message to log
     */
    log(message) {
        this._parent._log(message);
    }
    /**
     * Starts a performance counter.
     * @param counterName The name of the performance counter
     */
    startPerformanceCounter(counterName) {
        this._parent._startPerformanceCounter(counterName);
    }
    /**
     * Ends a performance counter.
     * @param counterName The name of the performance counter
     */
    endPerformanceCounter(counterName) {
        this._parent._endPerformanceCounter(counterName);
    }
}
/**
 * The default glTF sampler.
 */
GLTFLoader.DefaultSampler = { index: -1 };
GLTFFileLoader._CreateGLTF2Loader = (parent) => new GLTFLoader(parent);

export { ArrayItem, GLTFFileLoader, GLTFLoader, LoadBoundingInfoFromPositionAccessor };
//# sourceMappingURL=glTFLoader-BbG05oe0.js.map
