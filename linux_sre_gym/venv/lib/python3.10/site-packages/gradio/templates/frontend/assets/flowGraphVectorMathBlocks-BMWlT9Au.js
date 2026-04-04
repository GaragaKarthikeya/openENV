import { F as FlowGraphBlock, h as _GetClassNameOf } from './KHR_interactivity-BPhVhH5L.js';
import { e as RichTypeVector3, f as RichTypeQuaternion, b as RichTypeNumber, c as RichTypeBoolean, h as RichTypeMatrix, g as getRichTypeByFlowGraphType, l as RichTypeVector2, R as RichTypeAny } from './declarationMapper-eK_q7dJF.js';
import { ao as Quaternion, aB as Clamp, V as Vector3, ai as Vector4, R as RegisterClass } from './index-CxR0Yf_7.js';
import { F as FlowGraphBinaryOperationBlock } from './flowGraphBinaryOperationBlock-DJwjQkLd.js';
import { F as FlowGraphUnaryOperationBlock } from './flowGraphUnaryOperationBlock-BPV858Ax.js';
import './index-BvBk1Iap.js';
import './objectModelMapping-DnUo8NMz.js';
import './flowGraphCachedOperationBlock-HVWAtkJN.js';

/**
 * Creates a string representation of the IVector2Like
 * @param vector defines the IVector2Like to stringify
 * @param decimalCount defines the number of decimals to use
 * @returns a string with the IVector2Like coordinates.
 */
/**
 * Computes the dot product of two IVector3Like objects.
 * @param a defines the first vector
 * @param b defines the second vector
 * @returns the dot product
 */
function Vector3Dot(a, b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
/**
 * Computes the dot product of two IVector4Like objects
 * @param a defines the first vector
 * @param b defines the second vector
 * @returns the dot product
 */
function Vector4Dot(a, b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// *** NOTE ***
// These functions should ideally go in math.vector.functions.ts, but they require math.vector.ts to
// be imported which is big. To avoid the larger bundle size, they are kept inside flow graph for now.
/**
 * Returns the angle in radians between two quaternions
 * @param q1 defines the first quaternion
 * @param q2 defines the second quaternion
 * @returns the angle in radians between the two quaternions
 */
function GetAngleBetweenQuaternions(q1, q2) {
    return Math.acos(Clamp(Vector4Dot(q1, q2), -1, 1)) * 2;
}
/**
 * Creates a quaternion from two direction vectors
 * @param a defines the first direction vector
 * @param b defines the second direction vector
 * @returns the target quaternion
 */
function GetQuaternionFromDirections(a, b) {
    const result = new Quaternion();
    GetQuaternionFromDirectionsToRef(a, b, result);
    return result;
}
/**
 * Creates a quaternion from two direction vectors
 * @param a defines the first direction vector
 * @param b defines the second direction vector
 * @param result defines the target quaternion
 * @returns the target quaternion
 */
function GetQuaternionFromDirectionsToRef(a, b, result) {
    const axis = Vector3.Cross(a, b);
    const angle = Math.acos(Clamp(Vector3Dot(a, b), -1, 1));
    Quaternion.RotationAxisToRef(axis, angle, result);
    return result;
}

const AxisCacheName = "cachedOperationAxis";
const AngleCacheName = "cachedOperationAngle";
const CacheExecIdName = "cachedExecutionId";
/**
 * Vector length block.
 */
class FlowGraphLengthBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeNumber, (a) => this._polymorphicLength(a), "FlowGraphLengthBlock" /* FlowGraphBlockNames.Length */, config);
    }
    _polymorphicLength(a) {
        const aClassName = _GetClassNameOf(a);
        switch (aClassName) {
            case "Vector2" /* FlowGraphTypes.Vector2 */:
            case "Vector3" /* FlowGraphTypes.Vector3 */:
            case "Vector4" /* FlowGraphTypes.Vector4 */:
            case "Quaternion" /* FlowGraphTypes.Quaternion */:
                return a.length();
            default:
                throw new Error(`Cannot compute length of value ${a}`);
        }
    }
}
RegisterClass("FlowGraphLengthBlock" /* FlowGraphBlockNames.Length */, FlowGraphLengthBlock);
/**
 * Vector normalize block.
 */
class FlowGraphNormalizeBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeAny, (a) => this._polymorphicNormalize(a), "FlowGraphNormalizeBlock" /* FlowGraphBlockNames.Normalize */, config);
    }
    _polymorphicNormalize(a) {
        const aClassName = _GetClassNameOf(a);
        let normalized;
        switch (aClassName) {
            case "Vector2" /* FlowGraphTypes.Vector2 */:
            case "Vector3" /* FlowGraphTypes.Vector3 */:
            case "Vector4" /* FlowGraphTypes.Vector4 */:
            case "Quaternion" /* FlowGraphTypes.Quaternion */:
                normalized = a.normalizeToNew();
                if (this.config?.nanOnZeroLength) {
                    const length = a.length();
                    if (length === 0) {
                        normalized.setAll(NaN);
                    }
                }
                return normalized;
            default:
                throw new Error(`Cannot normalize value ${a}`);
        }
    }
}
RegisterClass("FlowGraphNormalizeBlock" /* FlowGraphBlockNames.Normalize */, FlowGraphNormalizeBlock);
/**
 * Dot product block.
 */
class FlowGraphDotBlock extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeAny, RichTypeNumber, (a, b) => this._polymorphicDot(a, b), "FlowGraphDotBlock" /* FlowGraphBlockNames.Dot */, config);
    }
    _polymorphicDot(a, b) {
        const className = _GetClassNameOf(a);
        switch (className) {
            case "Vector2" /* FlowGraphTypes.Vector2 */:
            case "Vector3" /* FlowGraphTypes.Vector3 */:
            case "Vector4" /* FlowGraphTypes.Vector4 */:
            case "Quaternion" /* FlowGraphTypes.Quaternion */:
                // casting is needed because dot requires both to be the same type
                return a.dot(b);
            default:
                throw new Error(`Cannot get dot product of ${a} and ${b}`);
        }
    }
}
RegisterClass("FlowGraphDotBlock" /* FlowGraphBlockNames.Dot */, FlowGraphDotBlock);
/**
 * Cross product block.
 */
class FlowGraphCrossBlock extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        super(RichTypeVector3, RichTypeVector3, RichTypeVector3, (a, b) => Vector3.Cross(a, b), "FlowGraphCrossBlock" /* FlowGraphBlockNames.Cross */, config);
    }
}
RegisterClass("FlowGraphCrossBlock" /* FlowGraphBlockNames.Cross */, FlowGraphCrossBlock);
/**
 * 2D rotation block.
 */
class FlowGraphRotate2DBlock extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        super(RichTypeVector2, RichTypeNumber, RichTypeVector2, (a, b) => a.rotate(b), "FlowGraphRotate2DBlock" /* FlowGraphBlockNames.Rotate2D */, config);
    }
}
RegisterClass("FlowGraphRotate2DBlock" /* FlowGraphBlockNames.Rotate2D */, FlowGraphRotate2DBlock);
/**
 * 3D rotation block.
 */
class FlowGraphRotate3DBlock extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        super(RichTypeVector3, RichTypeQuaternion, RichTypeVector3, (a, b) => a.applyRotationQuaternion(b), "FlowGraphRotate3DBlock" /* FlowGraphBlockNames.Rotate3D */, config);
    }
}
RegisterClass("FlowGraphRotate3DBlock" /* FlowGraphBlockNames.Rotate3D */, FlowGraphRotate3DBlock);
function TransformVector(a, b) {
    const className = _GetClassNameOf(a);
    switch (className) {
        case "Vector2" /* FlowGraphTypes.Vector2 */:
            return b.transformVector(a);
        case "Vector3" /* FlowGraphTypes.Vector3 */:
            return b.transformVector(a);
        case "Vector4" /* FlowGraphTypes.Vector4 */:
            a = a;
            // transform the vector 4 with the matrix here. Vector4.TransformCoordinates transforms a 3D coordinate, not Vector4
            return new Vector4(a.x * b.m[0] + a.y * b.m[1] + a.z * b.m[2] + a.w * b.m[3], a.x * b.m[4] + a.y * b.m[5] + a.z * b.m[6] + a.w * b.m[7], a.x * b.m[8] + a.y * b.m[9] + a.z * b.m[10] + a.w * b.m[11], a.x * b.m[12] + a.y * b.m[13] + a.z * b.m[14] + a.w * b.m[15]);
        default:
            throw new Error(`Cannot transform value ${a}`);
    }
}
/**
 * Transform a vector3 by a matrix.
 */
class FlowGraphTransformBlock extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        const vectorType = config?.vectorType || "Vector3" /* FlowGraphTypes.Vector3 */;
        const matrixType = vectorType === "Vector2" /* FlowGraphTypes.Vector2 */ ? "Matrix2D" /* FlowGraphTypes.Matrix2D */ : vectorType === "Vector3" /* FlowGraphTypes.Vector3 */ ? "Matrix3D" /* FlowGraphTypes.Matrix3D */ : "Matrix" /* FlowGraphTypes.Matrix */;
        super(getRichTypeByFlowGraphType(vectorType), getRichTypeByFlowGraphType(matrixType), getRichTypeByFlowGraphType(vectorType), TransformVector, "FlowGraphTransformVectorBlock" /* FlowGraphBlockNames.TransformVector */, config);
    }
}
RegisterClass("FlowGraphTransformVectorBlock" /* FlowGraphBlockNames.TransformVector */, FlowGraphTransformBlock);
/**
 * Transform a vector3 by a matrix.
 */
class FlowGraphTransformCoordinatesBlock extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        super(RichTypeVector3, RichTypeMatrix, RichTypeVector3, (a, b) => Vector3.TransformCoordinates(a, b), "FlowGraphTransformCoordinatesBlock" /* FlowGraphBlockNames.TransformCoordinates */, config);
    }
}
RegisterClass("FlowGraphTransformCoordinatesBlock" /* FlowGraphBlockNames.TransformCoordinates */, FlowGraphTransformCoordinatesBlock);
/**
 * Conjugate the quaternion.
 */
class FlowGraphConjugateBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeQuaternion, RichTypeQuaternion, (a) => a.conjugate(), "FlowGraphConjugateBlock" /* FlowGraphBlockNames.Conjugate */, config);
    }
}
RegisterClass("FlowGraphConjugateBlock" /* FlowGraphBlockNames.Conjugate */, FlowGraphConjugateBlock);
/**
 * Get the angle between two quaternions.
 */
class FlowGraphAngleBetweenBlock extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        super(RichTypeQuaternion, RichTypeQuaternion, RichTypeNumber, (a, b) => GetAngleBetweenQuaternions(a, b), "FlowGraphAngleBetweenBlock" /* FlowGraphBlockNames.AngleBetween */, config);
    }
}
RegisterClass("FlowGraphAngleBetweenBlock" /* FlowGraphBlockNames.AngleBetween */, FlowGraphAngleBetweenBlock);
/**
 * Get the quaternion from an axis and an angle.
 */
class FlowGraphQuaternionFromAxisAngleBlock extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        super(RichTypeVector3, RichTypeNumber, RichTypeQuaternion, (a, b) => Quaternion.RotationAxis(a, b), "FlowGraphQuaternionFromAxisAngleBlock" /* FlowGraphBlockNames.QuaternionFromAxisAngle */, config);
    }
}
RegisterClass("FlowGraphQuaternionFromAxisAngleBlock" /* FlowGraphBlockNames.QuaternionFromAxisAngle */, FlowGraphQuaternionFromAxisAngleBlock);
/**
 * Get the axis and angle from a quaternion.
 */
class FlowGraphAxisAngleFromQuaternionBlock extends FlowGraphBlock {
    constructor(config) {
        super(config);
        this.a = this.registerDataInput("a", RichTypeQuaternion);
        this.axis = this.registerDataOutput("axis", RichTypeVector3);
        this.angle = this.registerDataOutput("angle", RichTypeNumber);
        this.isValid = this.registerDataOutput("isValid", RichTypeBoolean);
    }
    /** @override */
    _updateOutputs(context) {
        const cachedExecutionId = context._getExecutionVariable(this, CacheExecIdName, -1);
        const cachedAxis = context._getExecutionVariable(this, AxisCacheName, null);
        const cachedAngle = context._getExecutionVariable(this, AngleCacheName, null);
        if (cachedAxis !== undefined && cachedAxis !== null && cachedAngle !== undefined && cachedAngle !== null && cachedExecutionId === context.executionId) {
            this.axis.setValue(cachedAxis, context);
            this.angle.setValue(cachedAngle, context);
        }
        else {
            try {
                const { axis, angle } = this.a.getValue(context).toAxisAngle();
                context._setExecutionVariable(this, AxisCacheName, axis);
                context._setExecutionVariable(this, AngleCacheName, angle);
                context._setExecutionVariable(this, CacheExecIdName, context.executionId);
                this.axis.setValue(axis, context);
                this.angle.setValue(angle, context);
                this.isValid.setValue(true, context);
            }
            catch (e) {
                this.isValid.setValue(false, context);
            }
        }
    }
    /** @override */
    getClassName() {
        return "FlowGraphAxisAngleFromQuaternionBlock" /* FlowGraphBlockNames.AxisAngleFromQuaternion */;
    }
}
RegisterClass("FlowGraphAxisAngleFromQuaternionBlock" /* FlowGraphBlockNames.AxisAngleFromQuaternion */, FlowGraphAxisAngleFromQuaternionBlock);
/**
 * Get the quaternion from two direction vectors.
 */
class FlowGraphQuaternionFromDirectionsBlock extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        super(RichTypeVector3, RichTypeVector3, RichTypeQuaternion, (a, b) => GetQuaternionFromDirections(a, b), "FlowGraphQuaternionFromDirectionsBlock" /* FlowGraphBlockNames.QuaternionFromDirections */, config);
    }
}

export { FlowGraphAngleBetweenBlock, FlowGraphAxisAngleFromQuaternionBlock, FlowGraphConjugateBlock, FlowGraphCrossBlock, FlowGraphDotBlock, FlowGraphLengthBlock, FlowGraphNormalizeBlock, FlowGraphQuaternionFromAxisAngleBlock, FlowGraphQuaternionFromDirectionsBlock, FlowGraphRotate2DBlock, FlowGraphRotate3DBlock, FlowGraphTransformBlock, FlowGraphTransformCoordinatesBlock };
//# sourceMappingURL=flowGraphVectorMathBlocks-BMWlT9Au.js.map
