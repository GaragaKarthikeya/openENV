import { i as RichTypeFlowGraphInteger, F as FlowGraphInteger, R as RichTypeAny } from './declarationMapper-eK_q7dJF.js';
import { C as Color3, b9 as Color4, V as Vector3, ai as Vector4, R as RegisterClass } from './index-CxR0Yf_7.js';
import { F as FlowGraphCachedOperationBlock } from './flowGraphCachedOperationBlock-HVWAtkJN.js';
import './index-BvBk1Iap.js';
import './KHR_interactivity-BPhVhH5L.js';
import './objectModelMapping-DnUo8NMz.js';

const PathHasTemplatesRegex = new RegExp(/\/\{(\w+)\}(?=\/|$)/g);
/**
 * @experimental
 * A component that converts a path to an object accessor.
 */
class FlowGraphPathConverterComponent {
    constructor(path, ownerBlock) {
        this.path = path;
        this.ownerBlock = ownerBlock;
        /**
         * The templated inputs for the provided path.
         */
        this.templatedInputs = [];
        let match = PathHasTemplatesRegex.exec(path);
        const templateSet = new Set();
        while (match) {
            const [, matchGroup] = match;
            if (templateSet.has(matchGroup)) {
                throw new Error("Duplicate template variable detected.");
            }
            templateSet.add(matchGroup);
            this.templatedInputs.push(ownerBlock.registerDataInput(matchGroup, RichTypeFlowGraphInteger, new FlowGraphInteger(0)));
            match = PathHasTemplatesRegex.exec(path);
        }
    }
    /**
     * Get the accessor for the path.
     * @param pathConverter the path converter to use to convert the path to an object accessor.
     * @param context the context to use.
     * @returns the accessor for the path.
     * @throws if the value for a templated input is invalid.
     */
    getAccessor(pathConverter, context) {
        let finalPath = this.path;
        for (const templatedInput of this.templatedInputs) {
            const valueToReplace = templatedInput.getValue(context).value;
            if (typeof valueToReplace !== "number" || valueToReplace < 0) {
                throw new Error("Invalid value for templated input.");
            }
            finalPath = finalPath.replace(`{${templatedInput.name}}`, valueToReplace.toString());
        }
        return pathConverter.convert(finalPath);
    }
}

/**
 * This block will take a JSON pointer and parse it to get the value from the JSON object.
 * The output is an object and a property name.
 * Optionally, the block can also output the value of the property. This is configurable.
 */
class FlowGraphJsonPointerParserBlock extends FlowGraphCachedOperationBlock {
    constructor(
    /**
     * the configuration of the block
     */
    config) {
        super(RichTypeAny, config);
        this.config = config;
        this.object = this.registerDataOutput("object", RichTypeAny);
        this.propertyName = this.registerDataOutput("propertyName", RichTypeAny);
        this.setterFunction = this.registerDataOutput("setFunction", RichTypeAny, this._setPropertyValue.bind(this));
        this.getterFunction = this.registerDataOutput("getFunction", RichTypeAny, this._getPropertyValue.bind(this));
        this.generateAnimationsFunction = this.registerDataOutput("generateAnimationsFunction", RichTypeAny, this._getInterpolationAnimationPropertyInfo.bind(this));
        this.templateComponent = new FlowGraphPathConverterComponent(config.jsonPointer, this);
    }
    _doOperation(context) {
        const accessorContainer = this.templateComponent.getAccessor(this.config.pathConverter, context);
        const value = accessorContainer.info.get(accessorContainer.object);
        const object = accessorContainer.info.getTarget?.(accessorContainer.object);
        const propertyName = accessorContainer.info.getPropertyName?.[0](accessorContainer.object);
        if (!object) {
            throw new Error("Object is undefined");
        }
        else {
            this.object.setValue(object, context);
            if (propertyName) {
                this.propertyName.setValue(propertyName, context);
            }
        }
        return value;
    }
    _setPropertyValue(_target, _propertyName, value, context) {
        const accessorContainer = this.templateComponent.getAccessor(this.config.pathConverter, context);
        const type = accessorContainer.info.type;
        if (type.startsWith("Color")) {
            value = ToColor(value, type);
        }
        accessorContainer.info.set?.(value, accessorContainer.object);
    }
    _getPropertyValue(_target, _propertyName, context) {
        const accessorContainer = this.templateComponent.getAccessor(this.config.pathConverter, context);
        const type = accessorContainer.info.type;
        const value = accessorContainer.info.get(accessorContainer.object);
        if (type.startsWith("Color")) {
            return FromColor(value);
        }
        return value;
    }
    _getInterpolationAnimationPropertyInfo(_target, _propertyName, context) {
        const accessorContainer = this.templateComponent.getAccessor(this.config.pathConverter, context);
        return (keys, fps, animationType, easingFunction) => {
            const animations = [];
            // make sure keys are of the right type (in case of float3 color/vector)
            const type = accessorContainer.info.type;
            if (type.startsWith("Color")) {
                keys = keys.map((key) => {
                    return {
                        frame: key.frame,
                        value: ToColor(key.value, type),
                    };
                });
            }
            accessorContainer.info.interpolation?.forEach((info, index) => {
                const name = accessorContainer.info.getPropertyName?.[index](accessorContainer.object) || "Animation-interpolation-" + index;
                // generate the keys based on interpolation info
                let newKeys = keys;
                if (animationType !== info.type) {
                    // convert the keys to the right type
                    newKeys = keys.map((key) => {
                        return {
                            frame: key.frame,
                            value: info.getValue(undefined, key.value.asArray ? key.value.asArray() : [key.value], 0, 1),
                        };
                    });
                }
                const animationData = info.buildAnimations(accessorContainer.object, name, 60, newKeys);
                for (const animation of animationData) {
                    if (easingFunction) {
                        animation.babylonAnimation.setEasingFunction(easingFunction);
                    }
                    animations.push(animation.babylonAnimation);
                }
            });
            return animations;
        };
    }
    /**
     * Gets the class name of this block
     * @returns the class name
     */
    getClassName() {
        return "FlowGraphJsonPointerParserBlock" /* FlowGraphBlockNames.JsonPointerParser */;
    }
}
function ToColor(value, expectedValue) {
    if (value.getClassName().startsWith("Color")) {
        return value;
    }
    if (expectedValue === "Color3") {
        return new Color3(value.x, value.y, value.z);
    }
    else if (expectedValue === "Color4") {
        return new Color4(value.x, value.y, value.z, value.w);
    }
    return value;
}
function FromColor(value) {
    if (value instanceof Color3) {
        return new Vector3(value.r, value.g, value.b);
    }
    else if (value instanceof Color4) {
        return new Vector4(value.r, value.g, value.b, value.a);
    }
    throw new Error("Invalid color type");
}
RegisterClass("FlowGraphJsonPointerParserBlock" /* FlowGraphBlockNames.JsonPointerParser */, FlowGraphJsonPointerParserBlock);

export { FlowGraphJsonPointerParserBlock };
//# sourceMappingURL=flowGraphJsonPointerParserBlock-BkiHU-aD.js.map
