import { ai as Vector4, M as Matrix, ao as Quaternion, V as Vector3, ah as Vector2, R as RegisterClass } from './index-CxR0Yf_7.js';
import { i as RichTypeFlowGraphInteger, F as FlowGraphInteger, g as getRichTypeByFlowGraphType, R as RichTypeAny, b as RichTypeNumber, c as RichTypeBoolean, n as FlowGraphMatrix2D, p as FlowGraphMatrix3D } from './declarationMapper-eK_q7dJF.js';
import { F as FlowGraphBinaryOperationBlock } from './flowGraphBinaryOperationBlock-DJwjQkLd.js';
import { F as FlowGraphCachedOperationBlock } from './flowGraphCachedOperationBlock-HVWAtkJN.js';
import { F as FlowGraphUnaryOperationBlock } from './flowGraphUnaryOperationBlock-BPV858Ax.js';
import { i as isNumeric, g as getNumericValue, h as _GetClassNameOf, j as _AreSameVectorOrQuaternionClass, k as _AreSameMatrixClass, l as _AreSameIntegerClass } from './KHR_interactivity-BPhVhH5L.js';
import './index-BvBk1Iap.js';
import './objectModelMapping-DnUo8NMz.js';

/**
 * Block that outputs a value of type ResultT, resulting of an operation with no inputs.
 * This block is being extended by some math operations and should not be used directly.
 * @internal
 */
class FlowGraphConstantOperationBlock extends FlowGraphCachedOperationBlock {
    constructor(richType, _operation, _className, config) {
        super(richType, config);
        this._operation = _operation;
        this._className = _className;
    }
    /**
     * the operation performed by this block
     * @param context the graph context
     * @returns the result of the operation
     */
    _doOperation(context) {
        return this._operation(context);
    }
    /**
     * Gets the class name of this block
     * @returns the class name
     */
    getClassName() {
        return this._className;
    }
}

/**
 * @internal
 * The base block for all ternary operation blocks.
 */
class FlowGraphTernaryOperationBlock extends FlowGraphCachedOperationBlock {
    constructor(t1Type, t2Type, t3Type, resultRichType, _operation, _className, config) {
        super(resultRichType, config);
        this._operation = _operation;
        this._className = _className;
        this.a = this.registerDataInput("a", t1Type);
        this.b = this.registerDataInput("b", t2Type);
        this.c = this.registerDataInput("c", t3Type);
    }
    /**
     * the operation performed by this block
     * @param context the graph context
     * @returns the result of the operation
     */
    _doOperation(context) {
        return this._operation(this.a.getValue(context), this.b.getValue(context), this.c.getValue(context));
    }
    /**
     * Gets the class name of this block
     * @returns the class name
     */
    getClassName() {
        return this._className;
    }
}

/**
 * Polymorphic add block.
 */
class FlowGraphAddBlock extends FlowGraphBinaryOperationBlock {
    /**
     * Construct a new add block.
     * @param config optional configuration
     */
    constructor(config) {
        super(getRichTypeByFlowGraphType(config?.type), getRichTypeByFlowGraphType(config?.type), getRichTypeByFlowGraphType(config?.type), (a, b) => this._polymorphicAdd(a, b), "FlowGraphAddBlock" /* FlowGraphBlockNames.Add */, config);
    }
    _polymorphicAdd(a, b) {
        const aClassName = _GetClassNameOf(a);
        const bClassName = _GetClassNameOf(b);
        if (_AreSameVectorOrQuaternionClass(aClassName, bClassName) || _AreSameMatrixClass(aClassName, bClassName) || _AreSameIntegerClass(aClassName, bClassName)) {
            // cast to vector3, but any other cast will be fine
            return a.add(b);
        }
        else if (aClassName === "Quaternion" /* FlowGraphTypes.Quaternion */ || bClassName === "Vector4" /* FlowGraphTypes.Vector4 */) {
            return new Vector4(a.x, a.y, a.z, a.w).addInPlace(b);
        }
        else if (aClassName === "Vector4" /* FlowGraphTypes.Vector4 */ || bClassName === "Quaternion" /* FlowGraphTypes.Quaternion */) {
            return a.add(b);
        }
        else {
            // at this point at least one of the variables is a number.
            if (this.config?.preventIntegerFloatArithmetic && typeof a !== typeof b) {
                throw new Error("Cannot add different types of numbers.");
            }
            return getNumericValue(a) + getNumericValue(b);
        }
    }
}
RegisterClass("FlowGraphAddBlock" /* FlowGraphBlockNames.Add */, FlowGraphAddBlock);
/**
 * Polymorphic subtract block.
 */
class FlowGraphSubtractBlock extends FlowGraphBinaryOperationBlock {
    /**
     * Construct a new subtract block.
     * @param config optional configuration
     */
    constructor(config) {
        super(getRichTypeByFlowGraphType(config?.type), getRichTypeByFlowGraphType(config?.type), getRichTypeByFlowGraphType(config?.type), (a, b) => this._polymorphicSubtract(a, b), "FlowGraphSubtractBlock" /* FlowGraphBlockNames.Subtract */, config);
    }
    _polymorphicSubtract(a, b) {
        const aClassName = _GetClassNameOf(a);
        const bClassName = _GetClassNameOf(b);
        if (_AreSameVectorOrQuaternionClass(aClassName, bClassName) || _AreSameIntegerClass(aClassName, bClassName) || _AreSameMatrixClass(aClassName, bClassName)) {
            // cast to vector3, but it can be casted to any vector type
            return a.subtract(b);
        }
        else if (aClassName === "Quaternion" /* FlowGraphTypes.Quaternion */ || bClassName === "Vector4" /* FlowGraphTypes.Vector4 */) {
            return new Vector4(a.x, a.y, a.z, a.w).subtractInPlace(b);
        }
        else if (aClassName === "Vector4" /* FlowGraphTypes.Vector4 */ || bClassName === "Quaternion" /* FlowGraphTypes.Quaternion */) {
            return a.subtract(b);
        }
        else {
            // at this point at least one of the variables is a number.
            if (this.config?.preventIntegerFloatArithmetic && typeof a !== typeof b) {
                throw new Error("Cannot add different types of numbers.");
            }
            return getNumericValue(a) - getNumericValue(b);
        }
    }
}
RegisterClass("FlowGraphSubtractBlock" /* FlowGraphBlockNames.Subtract */, FlowGraphSubtractBlock);
/**
 * Polymorphic multiply block.
 * In case of matrix, it is configurable whether the multiplication is done per component.
 */
class FlowGraphMultiplyBlock extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        super(getRichTypeByFlowGraphType(config?.type), getRichTypeByFlowGraphType(config?.type), getRichTypeByFlowGraphType(config?.type), (a, b) => this._polymorphicMultiply(a, b), "FlowGraphMultiplyBlock" /* FlowGraphBlockNames.Multiply */, config);
    }
    _polymorphicMultiply(a, b) {
        const aClassName = _GetClassNameOf(a);
        const bClassName = _GetClassNameOf(b);
        if (_AreSameVectorOrQuaternionClass(aClassName, bClassName) || _AreSameIntegerClass(aClassName, bClassName)) {
            // cast to vector3, but it can be casted to any vector type
            return a.multiply(b);
        }
        else if (aClassName === "Quaternion" /* FlowGraphTypes.Quaternion */ || bClassName === "Vector4" /* FlowGraphTypes.Vector4 */) {
            return new Vector4(a.x, a.y, a.z, a.w).multiplyInPlace(b);
        }
        else if (aClassName === "Vector4" /* FlowGraphTypes.Vector4 */ || bClassName === "Quaternion" /* FlowGraphTypes.Quaternion */) {
            return a.multiply(b);
        }
        else if (_AreSameMatrixClass(aClassName, bClassName)) {
            if (this.config?.useMatrixPerComponent) {
                // this is the definition of multiplication of glTF interactivity
                // get a's m as array, and multiply each component with b's m
                const aM = a.m;
                for (let i = 0; i < aM.length; i++) {
                    aM[i] *= b.m[i];
                }
                if (aClassName === "Matrix2D" /* FlowGraphTypes.Matrix2D */) {
                    return new FlowGraphMatrix2D(aM);
                }
                else if (aClassName === "Matrix3D" /* FlowGraphTypes.Matrix3D */) {
                    return new FlowGraphMatrix3D(aM);
                }
                else {
                    return Matrix.FromArray(aM);
                }
            }
            else {
                a = a;
                b = b;
                return b.multiply(a);
            }
        }
        else {
            // at this point at least one of the variables is a number.
            if (this.config?.preventIntegerFloatArithmetic && typeof a !== typeof b) {
                throw new Error("Cannot add different types of numbers.");
            }
            return getNumericValue(a) * getNumericValue(b);
        }
    }
}
RegisterClass("FlowGraphMultiplyBlock" /* FlowGraphBlockNames.Multiply */, FlowGraphMultiplyBlock);
/**
 * Polymorphic division block.
 */
class FlowGraphDivideBlock extends FlowGraphBinaryOperationBlock {
    /**
     * Construct a new divide block.
     * @param config - Optional configuration
     */
    constructor(config) {
        super(getRichTypeByFlowGraphType(config?.type), getRichTypeByFlowGraphType(config?.type), getRichTypeByFlowGraphType(config?.type), (a, b) => this._polymorphicDivide(a, b), "FlowGraphDivideBlock" /* FlowGraphBlockNames.Divide */, config);
    }
    _polymorphicDivide(a, b) {
        const aClassName = _GetClassNameOf(a);
        const bClassName = _GetClassNameOf(b);
        if (_AreSameVectorOrQuaternionClass(aClassName, bClassName) || _AreSameIntegerClass(aClassName, bClassName)) {
            // cast to vector3, but it can be casted to any vector type
            return a.divide(b);
        }
        else if (aClassName === "Quaternion" /* FlowGraphTypes.Quaternion */ || bClassName === "Quaternion" /* FlowGraphTypes.Quaternion */) {
            // this is a simple division (per component!), and should be also supported between Quat and Vector4. Therefore -
            const aClone = a.clone();
            aClone.x /= b.x;
            aClone.y /= b.y;
            aClone.z /= b.z;
            aClone.w /= b.w;
            return aClone;
        }
        else if (aClassName === "Quaternion" /* FlowGraphTypes.Quaternion */ || bClassName === "Vector4" /* FlowGraphTypes.Vector4 */) {
            return new Vector4(a.x, a.y, a.z, a.w).divideInPlace(b);
        }
        else if (aClassName === "Vector4" /* FlowGraphTypes.Vector4 */ || bClassName === "Quaternion" /* FlowGraphTypes.Quaternion */) {
            return a.divide(b);
        }
        else if (_AreSameMatrixClass(aClassName, bClassName)) {
            if (this.config?.useMatrixPerComponent) {
                // get a's m as array, and divide each component with b's m
                const aM = a.m;
                for (let i = 0; i < aM.length; i++) {
                    aM[i] /= b.m[i];
                }
                if (aClassName === "Matrix2D" /* FlowGraphTypes.Matrix2D */) {
                    return new FlowGraphMatrix2D(aM);
                }
                else if (aClassName === "Matrix3D" /* FlowGraphTypes.Matrix3D */) {
                    return new FlowGraphMatrix3D(aM);
                }
                else {
                    return Matrix.FromArray(aM);
                }
            }
            else {
                a = a;
                b = b;
                return a.divide(b);
            }
        }
        else {
            // at this point at least one of the variables is a number.
            if (this.config?.preventIntegerFloatArithmetic && typeof a !== typeof b) {
                throw new Error("Cannot add different types of numbers.");
            }
            return getNumericValue(a) / getNumericValue(b);
        }
    }
}
RegisterClass("FlowGraphDivideBlock" /* FlowGraphBlockNames.Divide */, FlowGraphDivideBlock);
/**
 * Random number between min and max (defaults to 0 to 1)
 *
 * This node will cache the result for he same node reference. i.e., a Math.eq that references the SAME random node will always return true.
 */
class FlowGraphRandomBlock extends FlowGraphConstantOperationBlock {
    /**
     * Construct a new random block.
     * @param config optional configuration
     */
    constructor(config) {
        super(RichTypeNumber, (context) => this._random(context), "FlowGraphRandomBlock" /* FlowGraphBlockNames.Random */, config);
        this.min = this.registerDataInput("min", RichTypeNumber, config?.min ?? 0);
        this.max = this.registerDataInput("max", RichTypeNumber, config?.max ?? 1);
        if (config?.seed) {
            this._seed = config.seed;
        }
    }
    _isSeed(seed = this._seed) {
        return seed !== undefined;
    }
    _getRandomValue() {
        if (this._isSeed(this._seed)) {
            // compute seed-based random number, deterministic randomness!
            const x = Math.sin(this._seed++) * 10000;
            return x - Math.floor(x);
        }
        return Math.random();
    }
    _random(context) {
        const min = this.min.getValue(context);
        const max = this.max.getValue(context);
        return this._getRandomValue() * (max - min) + min;
    }
}
RegisterClass("FlowGraphRandomBlock" /* FlowGraphBlockNames.Random */, FlowGraphRandomBlock);
/**
 * E constant.
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
class FlowGraphEBlock extends FlowGraphConstantOperationBlock {
    constructor(config) {
        super(RichTypeNumber, () => Math.E, "FlowGraphEBlock" /* FlowGraphBlockNames.E */, config);
    }
}
RegisterClass("FlowGraphEBlock" /* FlowGraphBlockNames.E */, FlowGraphEBlock);
/**
 * Pi constant.
 */
class FlowGraphPiBlock extends FlowGraphConstantOperationBlock {
    constructor(config) {
        super(RichTypeNumber, () => Math.PI, "FlowGraphPIBlock" /* FlowGraphBlockNames.PI */, config);
    }
}
RegisterClass("FlowGraphPIBlock" /* FlowGraphBlockNames.PI */, FlowGraphPiBlock);
/**
 * Positive inf constant.
 */
class FlowGraphInfBlock extends FlowGraphConstantOperationBlock {
    constructor(config) {
        super(RichTypeNumber, () => Number.POSITIVE_INFINITY, "FlowGraphInfBlock" /* FlowGraphBlockNames.Inf */, config);
    }
}
RegisterClass("FlowGraphInfBlock" /* FlowGraphBlockNames.Inf */, FlowGraphInfBlock);
/**
 * NaN constant.
 */
class FlowGraphNaNBlock extends FlowGraphConstantOperationBlock {
    constructor(config) {
        super(RichTypeNumber, () => Number.NaN, "FlowGraphNaNBlock" /* FlowGraphBlockNames.NaN */, config);
    }
}
RegisterClass("FlowGraphNaNBlock" /* FlowGraphBlockNames.NaN */, FlowGraphNaNBlock);
function ComponentWiseUnaryOperation(a, op) {
    const aClassName = _GetClassNameOf(a);
    switch (aClassName) {
        case "FlowGraphInteger":
            a = a;
            return new FlowGraphInteger(op(a.value));
        case "Vector2" /* FlowGraphTypes.Vector2 */:
            a = a;
            return new Vector2(op(a.x), op(a.y));
        case "Vector3" /* FlowGraphTypes.Vector3 */:
            a = a;
            return new Vector3(op(a.x), op(a.y), op(a.z));
        case "Vector4" /* FlowGraphTypes.Vector4 */:
            a = a;
            return new Vector4(op(a.x), op(a.y), op(a.z), op(a.w));
        case "Quaternion" /* FlowGraphTypes.Quaternion */:
            a = a;
            return new Quaternion(op(a.x), op(a.y), op(a.z), op(a.w));
        case "Matrix" /* FlowGraphTypes.Matrix */:
            a = a;
            return Matrix.FromArray(a.m.map(op));
        case "Matrix2D" /* FlowGraphTypes.Matrix2D */:
            a = a;
            // reason for not using .map is performance
            return new FlowGraphMatrix2D(a.m.map(op));
        case "Matrix3D" /* FlowGraphTypes.Matrix3D */:
            a = a;
            return new FlowGraphMatrix3D(a.m.map(op));
        default:
            a = a;
            return op(a);
    }
}
/**
 * Absolute value block.
 */
class FlowGraphAbsBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeNumber, RichTypeNumber, (a) => this._polymorphicAbs(a), "FlowGraphAbsBlock" /* FlowGraphBlockNames.Abs */, config);
    }
    _polymorphicAbs(a) {
        return ComponentWiseUnaryOperation(a, Math.abs);
    }
}
RegisterClass("FlowGraphAbsBlock" /* FlowGraphBlockNames.Abs */, FlowGraphAbsBlock);
/**
 * Sign block.
 */
class FlowGraphSignBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeNumber, RichTypeNumber, (a) => this._polymorphicSign(a), "FlowGraphSignBlock" /* FlowGraphBlockNames.Sign */, config);
    }
    _polymorphicSign(a) {
        return ComponentWiseUnaryOperation(a, Math.sign);
    }
}
RegisterClass("FlowGraphSignBlock" /* FlowGraphBlockNames.Sign */, FlowGraphSignBlock);
/**
 * Truncation block.
 */
class FlowGraphTruncBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeNumber, RichTypeNumber, (a) => this._polymorphicTrunc(a), "FlowGraphTruncBlock" /* FlowGraphBlockNames.Trunc */, config);
    }
    _polymorphicTrunc(a) {
        return ComponentWiseUnaryOperation(a, Math.trunc);
    }
}
RegisterClass("FlowGraphTruncBlock" /* FlowGraphBlockNames.Trunc */, FlowGraphTruncBlock);
/**
 * Floor block.
 */
class FlowGraphFloorBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeNumber, RichTypeNumber, (a) => this._polymorphicFloor(a), "FlowGraphFloorBlock" /* FlowGraphBlockNames.Floor */, config);
    }
    _polymorphicFloor(a) {
        return ComponentWiseUnaryOperation(a, Math.floor);
    }
}
RegisterClass("FlowGraphFloorBlock" /* FlowGraphBlockNames.Floor */, FlowGraphFloorBlock);
/**
 * Ceiling block.
 */
class FlowGraphCeilBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeAny, (a) => this._polymorphicCeiling(a), "FlowGraphCeilBlock" /* FlowGraphBlockNames.Ceil */, config);
    }
    _polymorphicCeiling(a) {
        return ComponentWiseUnaryOperation(a, Math.ceil);
    }
}
RegisterClass("FlowGraphCeilBlock" /* FlowGraphBlockNames.Ceil */, FlowGraphCeilBlock);
/**
 * Round block.
 */
class FlowGraphRoundBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeAny, (a) => this._polymorphicRound(a), "FlowGraphRoundBlock" /* FlowGraphBlockNames.Round */, config);
    }
    _polymorphicRound(a) {
        return ComponentWiseUnaryOperation(a, (a) => (a < 0 && this.config?.roundHalfAwayFromZero ? -Math.round(-a) : Math.round(a)));
    }
}
RegisterClass("FlowGraphRoundBlock" /* FlowGraphBlockNames.Round */, FlowGraphRoundBlock);
/**
 * A block that returns the fractional part of a number.
 */
class FlowGraphFractionBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeAny, (a) => this._polymorphicFraction(a), "FlowGraphFractBlock" /* FlowGraphBlockNames.Fraction */, config);
    }
    _polymorphicFraction(a) {
        return ComponentWiseUnaryOperation(a, (a) => a - Math.floor(a));
    }
}
RegisterClass("FlowGraphFractBlock" /* FlowGraphBlockNames.Fraction */, FlowGraphFractionBlock);
/**
 * Negation block.
 */
class FlowGraphNegationBlock extends FlowGraphUnaryOperationBlock {
    /**
     * construct a new negation block.
     * @param config optional configuration
     */
    constructor(config) {
        super(RichTypeAny, RichTypeAny, (a) => this._polymorphicNeg(a), "FlowGraphNegationBlock" /* FlowGraphBlockNames.Negation */, config);
    }
    _polymorphicNeg(a) {
        return ComponentWiseUnaryOperation(a, (a) => -a);
    }
}
RegisterClass("FlowGraphNegationBlock" /* FlowGraphBlockNames.Negation */, FlowGraphNegationBlock);
function ComponentWiseBinaryOperation(a, b, op) {
    const aClassName = _GetClassNameOf(a);
    switch (aClassName) {
        case "FlowGraphInteger":
            a = a;
            b = b;
            return new FlowGraphInteger(op(a.value, b.value));
        case "Vector2" /* FlowGraphTypes.Vector2 */:
            a = a;
            b = b;
            return new Vector2(op(a.x, b.x), op(a.y, b.y));
        case "Vector3" /* FlowGraphTypes.Vector3 */:
            a = a;
            b = b;
            return new Vector3(op(a.x, b.x), op(a.y, b.y), op(a.z, b.z));
        case "Vector4" /* FlowGraphTypes.Vector4 */:
            a = a;
            b = b;
            return new Vector4(op(a.x, b.x), op(a.y, b.y), op(a.z, b.z), op(a.w, b.w));
        case "Quaternion" /* FlowGraphTypes.Quaternion */:
            a = a;
            b = b;
            return new Quaternion(op(a.x, b.x), op(a.y, b.y), op(a.z, b.z), op(a.w, b.w));
        case "Matrix" /* FlowGraphTypes.Matrix */:
            a = a;
            return Matrix.FromArray(a.m.map((v, i) => op(v, b.m[i])));
        case "Matrix2D" /* FlowGraphTypes.Matrix2D */:
            a = a;
            return new FlowGraphMatrix2D(a.m.map((v, i) => op(v, b.m[i])));
        case "Matrix3D" /* FlowGraphTypes.Matrix3D */:
            a = a;
            return new FlowGraphMatrix3D(a.m.map((v, i) => op(v, b.m[i])));
        default:
            return op(getNumericValue(a), getNumericValue(b));
    }
}
/**
 * Remainder block.
 */
class FlowGraphModuloBlock extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeAny, RichTypeAny, (a, b) => this._polymorphicRemainder(a, b), "FlowGraphModuloBlock" /* FlowGraphBlockNames.Modulo */, config);
    }
    _polymorphicRemainder(a, b) {
        return ComponentWiseBinaryOperation(a, b, (a, b) => a % b);
    }
}
RegisterClass("FlowGraphModuloBlock" /* FlowGraphBlockNames.Modulo */, FlowGraphModuloBlock);
/**
 * Min block.
 */
class FlowGraphMinBlock extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeAny, RichTypeAny, (a, b) => this._polymorphicMin(a, b), "FlowGraphMinBlock" /* FlowGraphBlockNames.Min */, config);
    }
    _polymorphicMin(a, b) {
        return ComponentWiseBinaryOperation(a, b, Math.min);
    }
}
RegisterClass("FlowGraphMinBlock" /* FlowGraphBlockNames.Min */, FlowGraphMinBlock);
/**
 * Max block
 */
class FlowGraphMaxBlock extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeAny, RichTypeAny, (a, b) => this._polymorphicMax(a, b), "FlowGraphMaxBlock" /* FlowGraphBlockNames.Max */, config);
    }
    _polymorphicMax(a, b) {
        return ComponentWiseBinaryOperation(a, b, Math.max);
    }
}
RegisterClass("FlowGraphMaxBlock" /* FlowGraphBlockNames.Max */, FlowGraphMaxBlock);
function Clamp(a, b, c) {
    return Math.min(Math.max(a, Math.min(b, c)), Math.max(b, c));
}
function ComponentWiseTernaryOperation(a, b, c, op) {
    const aClassName = _GetClassNameOf(a);
    switch (aClassName) {
        case "FlowGraphInteger":
            a = a;
            b = b;
            c = c;
            return new FlowGraphInteger(op(a.value, b.value, c.value));
        case "Vector2" /* FlowGraphTypes.Vector2 */:
            a = a;
            b = b;
            c = c;
            return new Vector2(op(a.x, b.x, c.x), op(a.y, b.y, c.y));
        case "Vector3" /* FlowGraphTypes.Vector3 */:
            a = a;
            b = b;
            c = c;
            return new Vector3(op(a.x, b.x, c.x), op(a.y, b.y, c.y), op(a.z, b.z, c.z));
        case "Vector4" /* FlowGraphTypes.Vector4 */:
            a = a;
            b = b;
            c = c;
            return new Vector4(op(a.x, b.x, c.x), op(a.y, b.y, c.y), op(a.z, b.z, c.z), op(a.w, b.w, c.w));
        case "Quaternion" /* FlowGraphTypes.Quaternion */:
            a = a;
            b = b;
            c = c;
            return new Quaternion(op(a.x, b.x, c.x), op(a.y, b.y, c.y), op(a.z, b.z, c.z), op(a.w, b.w, c.w));
        case "Matrix" /* FlowGraphTypes.Matrix */:
            return Matrix.FromArray(a.m.map((v, i) => op(v, b.m[i], c.m[i])));
        case "Matrix2D" /* FlowGraphTypes.Matrix2D */:
            return new FlowGraphMatrix2D(a.m.map((v, i) => op(v, b.m[i], c.m[i])));
        case "Matrix3D" /* FlowGraphTypes.Matrix3D */:
            return new FlowGraphMatrix3D(a.m.map((v, i) => op(v, b.m[i], c.m[i])));
        default:
            return op(getNumericValue(a), getNumericValue(b), getNumericValue(c));
    }
}
/**
 * Clamp block.
 */
class FlowGraphClampBlock extends FlowGraphTernaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeAny, RichTypeAny, RichTypeAny, (a, b, c) => this._polymorphicClamp(a, b, c), "FlowGraphClampBlock" /* FlowGraphBlockNames.Clamp */, config);
    }
    _polymorphicClamp(a, b, c) {
        return ComponentWiseTernaryOperation(a, b, c, Clamp);
    }
}
RegisterClass("FlowGraphClampBlock" /* FlowGraphBlockNames.Clamp */, FlowGraphClampBlock);
function Saturate(a) {
    return Math.min(Math.max(a, 0), 1);
}
/**
 * Saturate block.
 */
class FlowGraphSaturateBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeAny, (a) => this._polymorphicSaturate(a), "FlowGraphSaturateBlock" /* FlowGraphBlockNames.Saturate */, config);
    }
    _polymorphicSaturate(a) {
        return ComponentWiseUnaryOperation(a, Saturate);
    }
}
RegisterClass("FlowGraphSaturateBlock" /* FlowGraphBlockNames.Saturate */, FlowGraphSaturateBlock);
function Interpolate(a, b, c) {
    return (1 - c) * a + c * b;
}
/**
 * Interpolate block.
 */
class FlowGraphMathInterpolationBlock extends FlowGraphTernaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeAny, RichTypeAny, RichTypeAny, (a, b, c) => this._polymorphicInterpolate(a, b, c), "FlowGraphMathInterpolationBlock" /* FlowGraphBlockNames.MathInterpolation */, config);
    }
    _polymorphicInterpolate(a, b, c) {
        return ComponentWiseTernaryOperation(a, b, c, Interpolate);
    }
}
RegisterClass("FlowGraphMathInterpolationBlock" /* FlowGraphBlockNames.MathInterpolation */, FlowGraphMathInterpolationBlock);
/**
 * Equals block.
 */
class FlowGraphEqualityBlock extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeAny, RichTypeBoolean, (a, b) => this._polymorphicEq(a, b), "FlowGraphEqualityBlock" /* FlowGraphBlockNames.Equality */, config);
    }
    _polymorphicEq(a, b) {
        const aClassName = _GetClassNameOf(a);
        const bClassName = _GetClassNameOf(b);
        if (typeof a !== typeof b) {
            return false;
        }
        if (_AreSameVectorOrQuaternionClass(aClassName, bClassName) || _AreSameMatrixClass(aClassName, bClassName) || _AreSameIntegerClass(aClassName, bClassName)) {
            return a.equals(b);
        }
        else {
            return a === b;
        }
    }
}
RegisterClass("FlowGraphEqualityBlock" /* FlowGraphBlockNames.Equality */, FlowGraphEqualityBlock);
function ComparisonOperators(a, b, op) {
    if (isNumeric(a) && isNumeric(b)) {
        return op(getNumericValue(a), getNumericValue(b));
    }
    else {
        throw new Error(`Cannot compare ${a} and ${b}`);
    }
}
/**
 * Less than block.
 */
class FlowGraphLessThanBlock extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeAny, RichTypeBoolean, (a, b) => this._polymorphicLessThan(a, b), "FlowGraphLessThanBlock" /* FlowGraphBlockNames.LessThan */, config);
    }
    _polymorphicLessThan(a, b) {
        return ComparisonOperators(a, b, (a, b) => a < b);
    }
}
RegisterClass("FlowGraphLessThanBlock" /* FlowGraphBlockNames.LessThan */, FlowGraphLessThanBlock);
/**
 * Less than or equal block.
 */
class FlowGraphLessThanOrEqualBlock extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeAny, RichTypeBoolean, (a, b) => this._polymorphicLessThanOrEqual(a, b), "FlowGraphLessThanOrEqualBlock" /* FlowGraphBlockNames.LessThanOrEqual */, config);
    }
    _polymorphicLessThanOrEqual(a, b) {
        return ComparisonOperators(a, b, (a, b) => a <= b);
    }
}
RegisterClass("FlowGraphLessThanOrEqualBlock" /* FlowGraphBlockNames.LessThanOrEqual */, FlowGraphLessThanOrEqualBlock);
/**
 * Greater than block.
 */
class FlowGraphGreaterThanBlock extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeAny, RichTypeBoolean, (a, b) => this._polymorphicGreaterThan(a, b), "FlowGraphGreaterThanBlock" /* FlowGraphBlockNames.GreaterThan */, config);
    }
    _polymorphicGreaterThan(a, b) {
        return ComparisonOperators(a, b, (a, b) => a > b);
    }
}
RegisterClass("FlowGraphGreaterThanBlock" /* FlowGraphBlockNames.GreaterThan */, FlowGraphGreaterThanBlock);
/**
 * Greater than or equal block.
 */
class FlowGraphGreaterThanOrEqualBlock extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeAny, RichTypeBoolean, (a, b) => this._polymorphicGreaterThanOrEqual(a, b), "FlowGraphGreaterThanOrEqualBlock" /* FlowGraphBlockNames.GreaterThanOrEqual */, config);
    }
    _polymorphicGreaterThanOrEqual(a, b) {
        return ComparisonOperators(a, b, (a, b) => a >= b);
    }
}
RegisterClass("FlowGraphGreaterThanOrEqualBlock" /* FlowGraphBlockNames.GreaterThanOrEqual */, FlowGraphGreaterThanOrEqualBlock);
/**
 * Is NaN block.
 */
class FlowGraphIsNanBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeBoolean, (a) => this._polymorphicIsNan(a), "FlowGraphIsNaNBlock" /* FlowGraphBlockNames.IsNaN */, config);
    }
    _polymorphicIsNan(a) {
        if (isNumeric(a, true)) {
            return isNaN(getNumericValue(a));
        }
        else {
            throw new Error(`Cannot get NaN of ${a}`);
        }
    }
}
RegisterClass("FlowGraphIsNaNBlock" /* FlowGraphBlockNames.IsNaN */, FlowGraphIsNanBlock);
/**
 * Is Inf block.
 */
class FlowGraphIsInfinityBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeBoolean, (a) => this._polymorphicIsInf(a), "FlowGraphIsInfBlock" /* FlowGraphBlockNames.IsInfinity */, config);
    }
    _polymorphicIsInf(a) {
        if (isNumeric(a)) {
            return !isFinite(getNumericValue(a));
        }
        else {
            throw new Error(`Cannot get isInf of ${a}`);
        }
    }
}
RegisterClass("FlowGraphIsInfBlock" /* FlowGraphBlockNames.IsInfinity */, FlowGraphIsInfinityBlock);
/**
 * Convert degrees to radians block.
 */
class FlowGraphDegToRadBlock extends FlowGraphUnaryOperationBlock {
    /**
     * Constructs a new instance of the flow graph math block.
     * @param config - Optional configuration for the flow graph block.
     */
    constructor(config) {
        super(RichTypeAny, RichTypeAny, (a) => this._polymorphicDegToRad(a), "FlowGraphDegToRadBlock" /* FlowGraphBlockNames.DegToRad */, config);
    }
    _degToRad(a) {
        return (a * Math.PI) / 180;
    }
    _polymorphicDegToRad(a) {
        return ComponentWiseUnaryOperation(a, this._degToRad);
    }
}
RegisterClass("FlowGraphDegToRadBlock" /* FlowGraphBlockNames.DegToRad */, FlowGraphDegToRadBlock);
/**
 * Convert radians to degrees block.
 */
class FlowGraphRadToDegBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeAny, (a) => this._polymorphicRadToDeg(a), "FlowGraphRadToDegBlock" /* FlowGraphBlockNames.RadToDeg */, config);
    }
    _radToDeg(a) {
        return (a * 180) / Math.PI;
    }
    _polymorphicRadToDeg(a) {
        return ComponentWiseUnaryOperation(a, this._radToDeg);
    }
}
RegisterClass("FlowGraphRadToDegBlock" /* FlowGraphBlockNames.RadToDeg */, FlowGraphRadToDegBlock);
/**
 * Sin block.
 */
class FlowGraphSinBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeNumber, RichTypeNumber, (a) => this._polymorphicSin(a), "FlowGraphSinBlock" /* FlowGraphBlockNames.Sin */, config);
    }
    _polymorphicSin(a) {
        return ComponentWiseUnaryOperation(a, Math.sin);
    }
}
/**
 * Cos block.
 */
class FlowGraphCosBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeNumber, RichTypeNumber, (a) => this._polymorphicCos(a), "FlowGraphCosBlock" /* FlowGraphBlockNames.Cos */, config);
    }
    _polymorphicCos(a) {
        return ComponentWiseUnaryOperation(a, Math.cos);
    }
}
/**
 * Tan block.
 */
class FlowGraphTanBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeNumber, RichTypeNumber, (a) => this._polymorphicTan(a), "FlowGraphTanBlock" /* FlowGraphBlockNames.Tan */, config);
    }
    _polymorphicTan(a) {
        return ComponentWiseUnaryOperation(a, Math.tan);
    }
}
/**
 * Arcsin block.
 */
class FlowGraphAsinBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeNumber, RichTypeNumber, (a) => this._polymorphicAsin(a), "FlowGraphASinBlock" /* FlowGraphBlockNames.Asin */, config);
    }
    _polymorphicAsin(a) {
        return ComponentWiseUnaryOperation(a, Math.asin);
    }
}
RegisterClass("FlowGraphASinBlock" /* FlowGraphBlockNames.Asin */, FlowGraphAsinBlock);
/**
 * Arccos block.
 */
class FlowGraphAcosBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeNumber, RichTypeNumber, (a) => this._polymorphicAcos(a), "FlowGraphACosBlock" /* FlowGraphBlockNames.Acos */, config);
    }
    _polymorphicAcos(a) {
        return ComponentWiseUnaryOperation(a, Math.acos);
    }
}
RegisterClass("FlowGraphACosBlock" /* FlowGraphBlockNames.Acos */, FlowGraphAcosBlock);
/**
 * Arctan block.
 */
class FlowGraphAtanBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeNumber, RichTypeNumber, (a) => this._polymorphicAtan(a), "FlowGraphATanBlock" /* FlowGraphBlockNames.Atan */, config);
    }
    _polymorphicAtan(a) {
        return ComponentWiseUnaryOperation(a, Math.atan);
    }
}
RegisterClass("FlowGraphATanBlock" /* FlowGraphBlockNames.Atan */, FlowGraphAtanBlock);
/**
 * Arctan2 block.
 */
class FlowGraphAtan2Block extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeAny, RichTypeAny, (a, b) => this._polymorphicAtan2(a, b), "FlowGraphATan2Block" /* FlowGraphBlockNames.Atan2 */, config);
    }
    _polymorphicAtan2(a, b) {
        return ComponentWiseBinaryOperation(a, b, Math.atan2);
    }
}
RegisterClass("FlowGraphATan2Block" /* FlowGraphBlockNames.Atan2 */, FlowGraphAtan2Block);
/**
 * Hyperbolic sin block.
 */
class FlowGraphSinhBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeAny, (a) => this._polymorphicSinh(a), "FlowGraphSinhBlock" /* FlowGraphBlockNames.Sinh */, config);
    }
    _polymorphicSinh(a) {
        return ComponentWiseUnaryOperation(a, Math.sinh);
    }
}
RegisterClass("FlowGraphSinhBlock" /* FlowGraphBlockNames.Sinh */, FlowGraphSinhBlock);
/**
 * Hyperbolic cos block.
 */
class FlowGraphCoshBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeAny, (a) => this._polymorphicCosh(a), "FlowGraphCoshBlock" /* FlowGraphBlockNames.Cosh */, config);
    }
    _polymorphicCosh(a) {
        return ComponentWiseUnaryOperation(a, Math.cosh);
    }
}
RegisterClass("FlowGraphCoshBlock" /* FlowGraphBlockNames.Cosh */, FlowGraphCoshBlock);
/**
 * Hyperbolic tan block.
 */
class FlowGraphTanhBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeAny, (a) => this._polymorphicTanh(a), "FlowGraphTanhBlock" /* FlowGraphBlockNames.Tanh */, config);
    }
    _polymorphicTanh(a) {
        return ComponentWiseUnaryOperation(a, Math.tanh);
    }
}
RegisterClass("FlowGraphTanhBlock" /* FlowGraphBlockNames.Tanh */, FlowGraphTanhBlock);
/**
 * Hyperbolic arcsin block.
 */
class FlowGraphAsinhBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeNumber, (a) => this._polymorphicAsinh(a), "FlowGraphASinhBlock" /* FlowGraphBlockNames.Asinh */, config);
    }
    _polymorphicAsinh(a) {
        return ComponentWiseUnaryOperation(a, Math.asinh);
    }
}
RegisterClass("FlowGraphASinhBlock" /* FlowGraphBlockNames.Asinh */, FlowGraphAsinhBlock);
/**
 * Hyperbolic arccos block.
 */
class FlowGraphAcoshBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeNumber, (a) => this._polymorphicAcosh(a), "FlowGraphACoshBlock" /* FlowGraphBlockNames.Acosh */, config);
    }
    _polymorphicAcosh(a) {
        return ComponentWiseUnaryOperation(a, Math.acosh);
    }
}
RegisterClass("FlowGraphACoshBlock" /* FlowGraphBlockNames.Acosh */, FlowGraphAcoshBlock);
/**
 * Hyperbolic arctan block.
 */
class FlowGraphAtanhBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeNumber, (a) => this._polymorphicAtanh(a), "FlowGraphATanhBlock" /* FlowGraphBlockNames.Atanh */, config);
    }
    _polymorphicAtanh(a) {
        return ComponentWiseUnaryOperation(a, Math.atanh);
    }
}
RegisterClass("FlowGraphATanhBlock" /* FlowGraphBlockNames.Atanh */, FlowGraphAtanhBlock);
/**
 * Exponential block.
 */
class FlowGraphExpBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeNumber, (a) => this._polymorphicExp(a), "FlowGraphExponentialBlock" /* FlowGraphBlockNames.Exponential */, config);
    }
    _polymorphicExp(a) {
        return ComponentWiseUnaryOperation(a, Math.exp);
    }
}
RegisterClass("FlowGraphExponentialBlock" /* FlowGraphBlockNames.Exponential */, FlowGraphExpBlock);
/**
 * Logarithm block.
 */
class FlowGraphLogBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeNumber, (a) => this._polymorphicLog(a), "FlowGraphLogBlock" /* FlowGraphBlockNames.Log */, config);
    }
    _polymorphicLog(a) {
        return ComponentWiseUnaryOperation(a, Math.log);
    }
}
RegisterClass("FlowGraphLogBlock" /* FlowGraphBlockNames.Log */, FlowGraphLogBlock);
/**
 * Base 2 logarithm block.
 */
class FlowGraphLog2Block extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeNumber, (a) => this._polymorphicLog2(a), "FlowGraphLog2Block" /* FlowGraphBlockNames.Log2 */, config);
    }
    _polymorphicLog2(a) {
        return ComponentWiseUnaryOperation(a, Math.log2);
    }
}
RegisterClass("FlowGraphLog2Block" /* FlowGraphBlockNames.Log2 */, FlowGraphLog2Block);
/**
 * Base 10 logarithm block.
 */
class FlowGraphLog10Block extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeNumber, (a) => this._polymorphicLog10(a), "FlowGraphLog10Block" /* FlowGraphBlockNames.Log10 */, config);
    }
    _polymorphicLog10(a) {
        return ComponentWiseUnaryOperation(a, Math.log10);
    }
}
RegisterClass("FlowGraphLog10Block" /* FlowGraphBlockNames.Log10 */, FlowGraphLog10Block);
/**
 * Square root block.
 */
class FlowGraphSquareRootBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeNumber, (a) => this._polymorphicSqrt(a), "FlowGraphSquareRootBlock" /* FlowGraphBlockNames.SquareRoot */, config);
    }
    _polymorphicSqrt(a) {
        return ComponentWiseUnaryOperation(a, Math.sqrt);
    }
}
RegisterClass("FlowGraphSquareRootBlock" /* FlowGraphBlockNames.SquareRoot */, FlowGraphSquareRootBlock);
/**
 * Cube root block.
 */
class FlowGraphCubeRootBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeNumber, (a) => this._polymorphicCubeRoot(a), "FlowGraphCubeRootBlock" /* FlowGraphBlockNames.CubeRoot */, config);
    }
    _polymorphicCubeRoot(a) {
        return ComponentWiseUnaryOperation(a, Math.cbrt);
    }
}
RegisterClass("FlowGraphCubeRootBlock" /* FlowGraphBlockNames.CubeRoot */, FlowGraphCubeRootBlock);
/**
 * Power block.
 */
class FlowGraphPowerBlock extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        super(RichTypeAny, RichTypeNumber, RichTypeNumber, (a, b) => this._polymorphicPow(a, b), "FlowGraphPowerBlock" /* FlowGraphBlockNames.Power */, config);
    }
    _polymorphicPow(a, b) {
        return ComponentWiseBinaryOperation(a, b, Math.pow);
    }
}
RegisterClass("FlowGraphPowerBlock" /* FlowGraphBlockNames.Power */, FlowGraphPowerBlock);
/**
 * Bitwise NOT operation
 */
class FlowGraphBitwiseNotBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(getRichTypeByFlowGraphType(config?.valueType || "FlowGraphInteger" /* FlowGraphTypes.Integer */), getRichTypeByFlowGraphType(config?.valueType || "FlowGraphInteger" /* FlowGraphTypes.Integer */), (a) => {
            if (typeof a === "boolean") {
                return !a;
            }
            else if (typeof a === "number") {
                return ~a;
            }
            return new FlowGraphInteger(~a.value);
        }, "FlowGraphBitwiseNotBlock" /* FlowGraphBlockNames.BitwiseNot */, config);
    }
}
RegisterClass("FlowGraphBitwiseNotBlock" /* FlowGraphBlockNames.BitwiseNot */, FlowGraphBitwiseNotBlock);
/**
 * Bitwise AND operation
 */
class FlowGraphBitwiseAndBlock extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        super(getRichTypeByFlowGraphType(config?.valueType || "FlowGraphInteger" /* FlowGraphTypes.Integer */), getRichTypeByFlowGraphType(config?.valueType || "FlowGraphInteger" /* FlowGraphTypes.Integer */), getRichTypeByFlowGraphType(config?.valueType || "FlowGraphInteger" /* FlowGraphTypes.Integer */), (a, b) => {
            if (typeof a === "boolean" && typeof b === "boolean") {
                return a && b;
            }
            else if (typeof a === "number" && typeof b === "number") {
                return a & b;
            }
            else if (typeof a === "object" && typeof b === "object") {
                return new FlowGraphInteger(a.value & b.value);
            }
            else {
                throw new Error(`Cannot perform bitwise AND on ${a} and ${b}`);
            }
        }, "FlowGraphBitwiseAndBlock" /* FlowGraphBlockNames.BitwiseAnd */, config);
    }
}
RegisterClass("FlowGraphBitwiseAndBlock" /* FlowGraphBlockNames.BitwiseAnd */, FlowGraphBitwiseAndBlock);
/**
 * Bitwise OR operation
 */
class FlowGraphBitwiseOrBlock extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        super(getRichTypeByFlowGraphType(config?.valueType || "FlowGraphInteger" /* FlowGraphTypes.Integer */), getRichTypeByFlowGraphType(config?.valueType || "FlowGraphInteger" /* FlowGraphTypes.Integer */), getRichTypeByFlowGraphType(config?.valueType || "FlowGraphInteger" /* FlowGraphTypes.Integer */), (a, b) => {
            if (typeof a === "boolean" && typeof b === "boolean") {
                return a || b;
            }
            else if (typeof a === "number" && typeof b === "number") {
                return a | b;
            }
            else if (typeof a === "object" && typeof b === "object") {
                return new FlowGraphInteger(a.value | b.value);
            }
            else {
                throw new Error(`Cannot perform bitwise OR on ${a} and ${b}`);
            }
        }, "FlowGraphBitwiseOrBlock" /* FlowGraphBlockNames.BitwiseOr */, config);
    }
}
RegisterClass("FlowGraphBitwiseOrBlock" /* FlowGraphBlockNames.BitwiseOr */, FlowGraphBitwiseOrBlock);
/**
 * Bitwise XOR operation
 */
class FlowGraphBitwiseXorBlock extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        super(getRichTypeByFlowGraphType(config?.valueType || "FlowGraphInteger" /* FlowGraphTypes.Integer */), getRichTypeByFlowGraphType(config?.valueType || "FlowGraphInteger" /* FlowGraphTypes.Integer */), getRichTypeByFlowGraphType(config?.valueType || "FlowGraphInteger" /* FlowGraphTypes.Integer */), (a, b) => {
            if (typeof a === "boolean" && typeof b === "boolean") {
                return a !== b;
            }
            else if (typeof a === "number" && typeof b === "number") {
                return a ^ b;
            }
            else if (typeof a === "object" && typeof b === "object") {
                return new FlowGraphInteger(a.value ^ b.value);
            }
            else {
                throw new Error(`Cannot perform bitwise XOR on ${a} and ${b}`);
            }
        }, "FlowGraphBitwiseXorBlock" /* FlowGraphBlockNames.BitwiseXor */, config);
    }
}
RegisterClass("FlowGraphBitwiseXorBlock" /* FlowGraphBlockNames.BitwiseXor */, FlowGraphBitwiseXorBlock);
/**
 * Bitwise left shift operation
 */
class FlowGraphBitwiseLeftShiftBlock extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        super(RichTypeFlowGraphInteger, RichTypeFlowGraphInteger, RichTypeFlowGraphInteger, (a, b) => new FlowGraphInteger(a.value << b.value), "FlowGraphBitwiseLeftShiftBlock" /* FlowGraphBlockNames.BitwiseLeftShift */, config);
    }
}
RegisterClass("FlowGraphBitwiseLeftShiftBlock" /* FlowGraphBlockNames.BitwiseLeftShift */, FlowGraphBitwiseLeftShiftBlock);
/**
 * Bitwise right shift operation
 */
class FlowGraphBitwiseRightShiftBlock extends FlowGraphBinaryOperationBlock {
    constructor(config) {
        super(RichTypeFlowGraphInteger, RichTypeFlowGraphInteger, RichTypeFlowGraphInteger, (a, b) => new FlowGraphInteger(a.value >> b.value), "FlowGraphBitwiseRightShiftBlock" /* FlowGraphBlockNames.BitwiseRightShift */, config);
    }
}
RegisterClass("FlowGraphBitwiseRightShiftBlock" /* FlowGraphBlockNames.BitwiseRightShift */, FlowGraphBitwiseRightShiftBlock);
/**
 * Count leading zeros operation
 */
class FlowGraphLeadingZerosBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeFlowGraphInteger, RichTypeFlowGraphInteger, (a) => new FlowGraphInteger(Math.clz32(a.value)), "FlowGraphLeadingZerosBlock" /* FlowGraphBlockNames.LeadingZeros */, config);
    }
}
RegisterClass("FlowGraphLeadingZerosBlock" /* FlowGraphBlockNames.LeadingZeros */, FlowGraphLeadingZerosBlock);
/**
 * Count trailing zeros operation
 */
class FlowGraphTrailingZerosBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeFlowGraphInteger, RichTypeFlowGraphInteger, (a) => new FlowGraphInteger(a.value ? 31 - Math.clz32(a.value & -a.value) : 32), "FlowGraphTrailingZerosBlock" /* FlowGraphBlockNames.TrailingZeros */, config);
    }
}
RegisterClass("FlowGraphTrailingZerosBlock" /* FlowGraphBlockNames.TrailingZeros */, FlowGraphTrailingZerosBlock);
/**
 * Given a number (which is converted to a 32-bit integer), return the
 * number of bits set to one on that number.
 * @param n the number to run the op on
 * @returns the number of bits set to one on that number
 */
function CountOnes(n) {
    let result = 0;
    while (n) {
        // This zeroes out all bits except for the least significant one.
        // So if the bit is set, it will be 1, otherwise it will be 0.
        result += n & 1;
        // This shifts n's bits to the right by one
        n >>= 1;
    }
    return result;
}
/**
 * Count one bits operation
 */
class FlowGraphOneBitsCounterBlock extends FlowGraphUnaryOperationBlock {
    constructor(config) {
        super(RichTypeFlowGraphInteger, RichTypeFlowGraphInteger, (a) => new FlowGraphInteger(CountOnes(a.value)), "FlowGraphOneBitsCounterBlock" /* FlowGraphBlockNames.OneBitsCounter */, config);
    }
}
RegisterClass("FlowGraphOneBitsCounterBlock" /* FlowGraphBlockNames.OneBitsCounter */, FlowGraphOneBitsCounterBlock);

export { FlowGraphAbsBlock, FlowGraphAcosBlock, FlowGraphAcoshBlock, FlowGraphAddBlock, FlowGraphAsinBlock, FlowGraphAsinhBlock, FlowGraphAtan2Block, FlowGraphAtanBlock, FlowGraphAtanhBlock, FlowGraphBitwiseAndBlock, FlowGraphBitwiseLeftShiftBlock, FlowGraphBitwiseNotBlock, FlowGraphBitwiseOrBlock, FlowGraphBitwiseRightShiftBlock, FlowGraphBitwiseXorBlock, FlowGraphCeilBlock, FlowGraphClampBlock, FlowGraphCosBlock, FlowGraphCoshBlock, FlowGraphCubeRootBlock, FlowGraphDegToRadBlock, FlowGraphDivideBlock, FlowGraphEBlock, FlowGraphEqualityBlock, FlowGraphExpBlock, FlowGraphFloorBlock, FlowGraphFractionBlock, FlowGraphGreaterThanBlock, FlowGraphGreaterThanOrEqualBlock, FlowGraphInfBlock, FlowGraphIsInfinityBlock, FlowGraphIsNanBlock, FlowGraphLeadingZerosBlock, FlowGraphLessThanBlock, FlowGraphLessThanOrEqualBlock, FlowGraphLog10Block, FlowGraphLog2Block, FlowGraphLogBlock, FlowGraphMathInterpolationBlock, FlowGraphMaxBlock, FlowGraphMinBlock, FlowGraphModuloBlock, FlowGraphMultiplyBlock, FlowGraphNaNBlock, FlowGraphNegationBlock, FlowGraphOneBitsCounterBlock, FlowGraphPiBlock, FlowGraphPowerBlock, FlowGraphRadToDegBlock, FlowGraphRandomBlock, FlowGraphRoundBlock, FlowGraphSaturateBlock, FlowGraphSignBlock, FlowGraphSinBlock, FlowGraphSinhBlock, FlowGraphSquareRootBlock, FlowGraphSubtractBlock, FlowGraphTanBlock, FlowGraphTanhBlock, FlowGraphTrailingZerosBlock, FlowGraphTruncBlock };
//# sourceMappingURL=flowGraphMathBlocks-CBHd7jx3.js.map
