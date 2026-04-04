/* IMPORT */
/* MAIN */
const cloneDeep = (value) => {
    return JSON.parse(JSON.stringify(value));
};
const isElement = (value) => {
    return (value.nodeType === 1);
};
const isElementFunky = (value) => {
    return FUNKY_TAG_NAMES.has(value.tagName);
};
const isElementAction = (value) => {
    return ('action' in value);
};
const isElementIframe = (value) => {
    return (value.tagName === 'IFRAME');
};
const isElementFormAction = (value) => {
    return ('formAction' in value);
};
const isElementHyperlink = (value) => {
    return ('protocol' in value);
};
const isScriptOrDataUrl = (() => {
    const re = /^(?:\w+script|data):/i;
    return (url) => {
        return re.test(url);
    };
})();
const isScriptOrDataUrlLoose = (() => {
    const re = /(?:script|data):/i;
    return (url) => {
        return re.test(url);
    };
})();
const mergeMaps = (maps) => {
    const merged = {};
    for (let i = 0, l = maps.length; i < l; i++) {
        const map = maps[i];
        for (const key in map) {
            if (!merged[key]) {
                merged[key] = map[key];
            }
            else {
                merged[key] = merged[key].concat(map[key]);
            }
        }
    }
    return merged;
};
const traverseElementsBasic = (parent, callback) => {
    let current = parent.firstChild;
    while (current) {
        const next = current.nextSibling;
        if (isElement(current)) {
            callback(current, parent);
            if (current.parentNode) { // Still connected, so recurse
                traverseElementsBasic(current, callback);
            }
        }
        current = next;
    }
};
const traverseElementsIterator = (parent, callback) => {
    const iterator = document.createNodeIterator(parent, NodeFilter.SHOW_ELEMENT);
    let current;
    while (current = iterator.nextNode()) {
        const parent = current.parentNode;
        if (!parent)
            continue;
        callback(current, parent); //TSC
    }
};
const traverseElements = (parent, callback) => {
    const hasIterator = !!globalThis.document && !!globalThis.document.createNodeIterator; // For better WebWorker support
    if (hasIterator) {
        return traverseElementsIterator(parent, callback);
    }
    else {
        return traverseElementsBasic(parent, callback);
    }
};

/* IMPORT */
/* ELEMENTS */
const HTML_ELEMENTS_ALLOW = [
    'a',
    'abbr',
    'acronym',
    'address',
    'area',
    'article',
    'aside',
    'audio',
    'b',
    'bdi',
    'bdo',
    'bgsound',
    'big',
    'blockquote',
    'body',
    'br',
    'button',
    'canvas',
    'caption',
    'center',
    'cite',
    'code',
    'col',
    'colgroup',
    'datalist',
    'dd',
    'del',
    'details',
    'dfn',
    'dialog',
    'dir',
    'div',
    'dl',
    'dt',
    'em',
    'fieldset',
    'figcaption',
    'figure',
    'font',
    'footer',
    'form',
    'h1',
    'h2',
    'h3',
    'h4',
    'h5',
    'h6',
    'head',
    'header',
    'hgroup',
    'hr',
    'html',
    'i',
    'img',
    'input',
    'ins',
    'kbd',
    'keygen',
    'label',
    'layer',
    'legend',
    'li',
    'link',
    'listing',
    'main',
    'map',
    'mark',
    'marquee',
    'menu',
    'meta',
    'meter',
    'nav',
    'nobr',
    'ol',
    'optgroup',
    'option',
    'output',
    'p',
    'picture',
    'popup',
    'pre',
    'progress',
    'q',
    'rb',
    'rp',
    'rt',
    'rtc',
    'ruby',
    's',
    'samp',
    'section',
    'select',
    'selectmenu',
    'small',
    'source',
    'span',
    'strike',
    'strong',
    'style',
    'sub',
    'summary',
    'sup',
    'table',
    'tbody',
    'td',
    'tfoot',
    'th',
    'thead',
    'time',
    'tr',
    'track',
    'tt',
    'u',
    'ul',
    'var',
    'video',
    'wbr'
];
const HTML_ELEMENTS_DISALLOW = [
    'basefont',
    'command',
    'data',
    'iframe',
    'image',
    'plaintext',
    'portal',
    'slot',
    // 'template', //TODO: Not exactly correct to never allow this, too strict
    'textarea',
    'title',
    'xmp'
];
const HTML_ELEMENTS = new Set([
    ...HTML_ELEMENTS_ALLOW,
    ...HTML_ELEMENTS_DISALLOW
]);
const SVG_ELEMENTS_ALLOW = [
    'svg',
    'a',
    'altglyph',
    'altglyphdef',
    'altglyphitem',
    'animatecolor',
    'animatemotion',
    'animatetransform',
    'circle',
    'clippath',
    'defs',
    'desc',
    'ellipse',
    'filter',
    'font',
    'g',
    'glyph',
    'glyphref',
    'hkern',
    'image',
    'line',
    'lineargradient',
    'marker',
    'mask',
    'metadata',
    'mpath',
    'path',
    'pattern',
    'polygon',
    'polyline',
    'radialgradient',
    'rect',
    'stop',
    'style',
    'switch',
    'symbol',
    'text',
    'textpath',
    'title',
    'tref',
    'tspan',
    'view',
    'vkern',
    /* FILTERS */
    'feBlend',
    'feColorMatrix',
    'feComponentTransfer',
    'feComposite',
    'feConvolveMatrix',
    'feDiffuseLighting',
    'feDisplacementMap',
    'feDistantLight',
    'feFlood',
    'feFuncA',
    'feFuncB',
    'feFuncG',
    'feFuncR',
    'feGaussianBlur',
    'feImage',
    'feMerge',
    'feMergeNode',
    'feMorphology',
    'feOffset',
    'fePointLight',
    'feSpecularLighting',
    'feSpotLight',
    'feTile',
    'feTurbulence'
];
const SVG_ELEMENTS_DISALLOW = [
    'animate',
    'color-profile',
    'cursor',
    'discard',
    'fedropshadow',
    'font-face',
    'font-face-format',
    'font-face-name',
    'font-face-src',
    'font-face-uri',
    'foreignobject',
    'hatch',
    'hatchpath',
    'mesh',
    'meshgradient',
    'meshpatch',
    'meshrow',
    'missing-glyph',
    'script',
    'set',
    'solidcolor',
    'unknown',
    'use'
];
const SVG_ELEMENTS = new Set([
    ...SVG_ELEMENTS_ALLOW,
    ...SVG_ELEMENTS_DISALLOW
]);
const MATH_ELEMENTS_ALLOW = [
    'math',
    'menclose',
    'merror',
    'mfenced',
    'mfrac',
    'mglyph',
    'mi',
    'mlabeledtr',
    'mmultiscripts',
    'mn',
    'mo',
    'mover',
    'mpadded',
    'mphantom',
    'mroot',
    'mrow',
    'ms',
    'mspace',
    'msqrt',
    'mstyle',
    'msub',
    'msup',
    'msubsup',
    'mtable',
    'mtd',
    'mtext',
    'mtr',
    'munder',
    'munderover'
];
const MATH_ELEMENTS_DISALLOW = [
    'maction',
    'maligngroup',
    'malignmark',
    'mlongdiv',
    'mscarries',
    'mscarry',
    'msgroup',
    'mstack',
    'msline',
    'msrow',
    'semantics',
    'annotation',
    'annotation-xml',
    'mprescripts',
    'none'
];
const MATH_ELEMENTS = new Set([
    ...MATH_ELEMENTS_ALLOW,
    ...MATH_ELEMENTS_DISALLOW
]);
/* ATTRIBUTES */
const HTML_ATTRIBUTES_ALLOW = [
    'abbr',
    'accept',
    'accept-charset',
    'accesskey',
    'action',
    'align',
    'alink',
    'allow',
    'allowfullscreen',
    'alt',
    'anchor',
    'archive',
    'as',
    'async',
    'autocapitalize',
    'autocomplete',
    'autocorrect',
    'autofocus',
    'autopictureinpicture',
    'autoplay',
    'axis',
    'background',
    'behavior',
    'bgcolor',
    'border',
    'bordercolor',
    'capture',
    'cellpadding',
    'cellspacing',
    'challenge',
    'char',
    'charoff',
    'charset',
    'checked',
    'cite',
    'class',
    'classid',
    'clear',
    'code',
    'codebase',
    'codetype',
    'color',
    'cols',
    'colspan',
    'compact',
    'content',
    'contenteditable',
    'controls',
    'controlslist',
    'conversiondestination',
    'coords',
    'crossorigin',
    'csp',
    'data',
    'datetime',
    'declare',
    'decoding',
    'default',
    'defer',
    'dir',
    'direction',
    'dirname',
    'disabled',
    'disablepictureinpicture',
    'disableremoteplayback',
    'disallowdocumentaccess',
    'download',
    'draggable',
    'elementtiming',
    'enctype',
    'end',
    'enterkeyhint',
    'event',
    'exportparts',
    'face',
    'for',
    'form',
    'formaction',
    'formenctype',
    'formmethod',
    'formnovalidate',
    'formtarget',
    'frame',
    'frameborder',
    'headers',
    'height',
    'hidden',
    'high',
    'href',
    'hreflang',
    'hreftranslate',
    'hspace',
    'http-equiv',
    'id',
    'imagesizes',
    'imagesrcset',
    'importance',
    'impressiondata',
    'impressionexpiry',
    'incremental',
    'inert',
    'inputmode',
    'integrity',
    'invisible',
    'ismap',
    'keytype',
    'kind',
    'label',
    'lang',
    'language',
    'latencyhint',
    'leftmargin',
    'link',
    'list',
    'loading',
    'longdesc',
    'loop',
    'low',
    'lowsrc',
    'manifest',
    'marginheight',
    'marginwidth',
    'max',
    'maxlength',
    'mayscript',
    'media',
    'method',
    'min',
    'minlength',
    'multiple',
    'muted',
    'name',
    'nohref',
    'nomodule',
    'nonce',
    'noresize',
    'noshade',
    'novalidate',
    'nowrap',
    'object',
    'open',
    'optimum',
    'part',
    'pattern',
    'ping',
    'placeholder',
    'playsinline',
    'policy',
    'poster',
    'preload',
    'pseudo',
    'readonly',
    'referrerpolicy',
    'rel',
    'reportingorigin',
    'required',
    'resources',
    'rev',
    'reversed',
    'role',
    'rows',
    'rowspan',
    'rules',
    'sandbox',
    'scheme',
    'scope',
    'scopes',
    'scrollamount',
    'scrolldelay',
    'scrolling',
    'select',
    'selected',
    'shadowroot',
    'shadowrootdelegatesfocus',
    'shape',
    'size',
    'sizes',
    'slot',
    'span',
    'spellcheck',
    'src',
    'srclang',
    'srcset',
    'standby',
    'start',
    'step',
    'style',
    'summary',
    'tabindex',
    'target',
    'text',
    'title',
    'topmargin',
    'translate',
    'truespeed',
    'trusttoken',
    'type',
    'usemap',
    'valign',
    'value',
    'valuetype',
    'version',
    'virtualkeyboardpolicy',
    'vlink',
    'vspace',
    'webkitdirectory',
    'width',
    'wrap'
];
const SVG_ATTRIBUTES_ALLOW = [
    'accent-height',
    'accumulate',
    'additive',
    'alignment-baseline',
    'ascent',
    'attributename',
    'attributetype',
    'azimuth',
    'basefrequency',
    'baseline-shift',
    'begin',
    'bias',
    'by',
    'class',
    'clip',
    'clippathunits',
    'clip-path',
    'clip-rule',
    'color',
    'color-interpolation',
    'color-interpolation-filters',
    'color-profile',
    'color-rendering',
    'cx',
    'cy',
    'd',
    'dx',
    'dy',
    'diffuseconstant',
    'direction',
    'display',
    'divisor',
    'dominant-baseline',
    'dur',
    'edgemode',
    'elevation',
    'end',
    'fill',
    'fill-opacity',
    'fill-rule',
    'filter',
    'filterunits',
    'flood-color',
    'flood-opacity',
    'font-family',
    'font-size',
    'font-size-adjust',
    'font-stretch',
    'font-style',
    'font-variant',
    'font-weight',
    'fx',
    'fy',
    'g1',
    'g2',
    'glyph-name',
    'glyphref',
    'gradientunits',
    'gradienttransform',
    'height',
    'href',
    'id',
    'image-rendering',
    'in',
    'in2',
    'k',
    'k1',
    'k2',
    'k3',
    'k4',
    'kerning',
    'keypoints',
    'keysplines',
    'keytimes',
    'lang',
    'lengthadjust',
    'letter-spacing',
    'kernelmatrix',
    'kernelunitlength',
    'lighting-color',
    'local',
    'marker-end',
    'marker-mid',
    'marker-start',
    'markerheight',
    'markerunits',
    'markerwidth',
    'maskcontentunits',
    'maskunits',
    'max',
    'mask',
    'media',
    'method',
    'mode',
    'min',
    'name',
    'numoctaves',
    'offset',
    'operator',
    'opacity',
    'order',
    'orient',
    'orientation',
    'origin',
    'overflow',
    'paint-order',
    'path',
    'pathlength',
    'patterncontentunits',
    'patterntransform',
    'patternunits',
    'points',
    'preservealpha',
    'preserveaspectratio',
    'primitiveunits',
    'r',
    'rx',
    'ry',
    'radius',
    'refx',
    'refy',
    'repeatcount',
    'repeatdur',
    'restart',
    'result',
    'rotate',
    'scale',
    'seed',
    'shape-rendering',
    'specularconstant',
    'specularexponent',
    'spreadmethod',
    'startoffset',
    'stddeviation',
    'stitchtiles',
    'stop-color',
    'stop-opacity',
    'stroke-dasharray',
    'stroke-dashoffset',
    'stroke-linecap',
    'stroke-linejoin',
    'stroke-miterlimit',
    'stroke-opacity',
    'stroke',
    'stroke-width',
    'style',
    'surfacescale',
    'systemlanguage',
    'tabindex',
    'targetx',
    'targety',
    'transform',
    'transform-origin',
    'text-anchor',
    'text-decoration',
    'text-rendering',
    'textlength',
    'type',
    'u1',
    'u2',
    'unicode',
    'values',
    'viewbox',
    'visibility',
    'version',
    'vert-adv-y',
    'vert-origin-x',
    'vert-origin-y',
    'width',
    'word-spacing',
    'wrap',
    'writing-mode',
    'xchannelselector',
    'ychannelselector',
    'x',
    'x1',
    'x2',
    'xmlns',
    'y',
    'y1',
    'y2',
    'z',
    'zoomandpan'
];
const MATH_ATTRIBUTES_ALLOW = [
    'accent',
    'accentunder',
    'align',
    'bevelled',
    'close',
    'columnsalign',
    'columnlines',
    'columnspan',
    'denomalign',
    'depth',
    'dir',
    'display',
    'displaystyle',
    'encoding',
    'fence',
    'frame',
    'height',
    'href',
    'id',
    'largeop',
    'length',
    'linethickness',
    'lspace',
    'lquote',
    'mathbackground',
    'mathcolor',
    'mathsize',
    'mathvariant',
    'maxsize',
    'minsize',
    'movablelimits',
    'notation',
    'numalign',
    'open',
    'rowalign',
    'rowlines',
    'rowspacing',
    'rowspan',
    'rspace',
    'rquote',
    'scriptlevel',
    'scriptminsize',
    'scriptsizemultiplier',
    'selection',
    'separator',
    'separators',
    'stretchy',
    'subscriptshift',
    'supscriptshift',
    'symmetric',
    'voffset',
    'width',
    'xmlns'
];
/* NAMESPACES */
const NAMESPACES = {
    HTML: 'http://www.w3.org/1999/xhtml',
    SVG: 'http://www.w3.org/2000/svg',
    MATH: 'http://www.w3.org/1998/Math/MathML'
};
const NAMESPACES_ELEMENTS = {
    [NAMESPACES.HTML]: HTML_ELEMENTS,
    [NAMESPACES.SVG]: SVG_ELEMENTS,
    [NAMESPACES.MATH]: MATH_ELEMENTS
};
const NAMESPACES_ROOTS = {
    [NAMESPACES.HTML]: 'html',
    [NAMESPACES.SVG]: 'svg',
    [NAMESPACES.MATH]: 'math'
};
const NAMESPACES_PREFIXES = {
    [NAMESPACES.HTML]: '',
    [NAMESPACES.SVG]: 'svg:',
    [NAMESPACES.MATH]: 'math:'
};
/* TAG NAMES */
const FUNKY_TAG_NAMES = new Set([
    'A',
    'AREA',
    'BUTTON',
    'FORM',
    'IFRAME',
    'INPUT'
]);
/* OTHERS */
const DEFAULTS = {
    allowComments: true,
    allowCustomElements: false,
    allowUnknownMarkup: false,
    allowElements: [
        ...HTML_ELEMENTS_ALLOW,
        ...SVG_ELEMENTS_ALLOW.map(name => `svg:${name}`),
        ...MATH_ELEMENTS_ALLOW.map(name => `math:${name}`)
    ],
    allowAttributes: mergeMaps([
        Object.fromEntries(HTML_ATTRIBUTES_ALLOW.map(name => [name, ['*']])),
        Object.fromEntries(SVG_ATTRIBUTES_ALLOW.map(name => [name, ['svg:*']])),
        Object.fromEntries(MATH_ATTRIBUTES_ALLOW.map(name => [name, ['math:*']]))
    ])
};

/* IMPORT */
var __classPrivateFieldSet = (undefined && undefined.__classPrivateFieldSet) || function (receiver, state, value, kind, f) {
    if (kind === "m") throw new TypeError("Private method is not writable");
    if (kind === "a" && !f) throw new TypeError("Private accessor was defined without a setter");
    if (typeof state === "function" ? receiver !== state || !f : !state.has(receiver)) throw new TypeError("Cannot write private member to an object whose class did not declare it");
    return (kind === "a" ? f.call(receiver, value) : f ? f.value = value : state.set(receiver, value)), value;
};
var __classPrivateFieldGet = (undefined && undefined.__classPrivateFieldGet) || function (receiver, state, kind, f) {
    if (kind === "a" && !f) throw new TypeError("Private accessor was defined without a getter");
    if (typeof state === "function" ? receiver !== state || !f : !state.has(receiver)) throw new TypeError("Cannot read private member from an object whose class did not declare it");
    return kind === "m" ? f : kind === "a" ? f.call(receiver) : f ? f.value : state.get(receiver);
};
var _Amuchina_configuration, _Amuchina_allowElements, _Amuchina_allowAttributes;
/* MAIN */
//TODO: Add a decent test suite, possibly one from an existing trusted library
class Amuchina {
    /* CONSTRUCTOR */
    constructor(configuration = {}) {
        /* VARIABLES */
        _Amuchina_configuration.set(this, void 0);
        _Amuchina_allowElements.set(this, void 0);
        _Amuchina_allowAttributes.set(this, void 0);
        /* API */
        this.getConfiguration = () => {
            return cloneDeep(__classPrivateFieldGet(this, _Amuchina_configuration, "f"));
        };
        this.sanitize = (input) => {
            //TODO: Support integration points (foreignObject and friends)
            //TODO: Support xlink:href, xml:id, xlink:title, xml:space, xmlns:xlink
            const allowElements = __classPrivateFieldGet(this, _Amuchina_allowElements, "f");
            const allowAttributes = __classPrivateFieldGet(this, _Amuchina_allowAttributes, "f");
            traverseElements(input, (node, parent) => {
                const namespace = node.namespaceURI || NAMESPACES.HTML;
                const namespaceParent = parent['namespaceURI'] || NAMESPACES.HTML;
                const elements = NAMESPACES_ELEMENTS[namespace];
                const root = NAMESPACES_ROOTS[namespace];
                const prefix = NAMESPACES_PREFIXES[namespace];
                const tag = node.tagName.toLowerCase();
                const tagPrefixed = `${prefix}${tag}`;
                const all = '*';
                const allPrefixed = `${prefix}${all}`;
                if (!elements.has(tag) || !allowElements.has(tagPrefixed) || (namespace !== namespaceParent && tag !== root)) {
                    parent.removeChild(node);
                }
                else {
                    const attributes = node.getAttributeNames();
                    const attributesLength = attributes.length;
                    if (attributesLength) {
                        for (let i = 0; i < attributesLength; i++) {
                            const attribute = attributes[i];
                            const allowedValues = allowAttributes[attribute];
                            if (!allowedValues || (!allowedValues.has(allPrefixed) && !allowedValues.has(tagPrefixed))) {
                                node.removeAttribute(attribute);
                            }
                        }
                        if (isElementFunky(node)) {
                            if (isElementHyperlink(node)) {
                                const href = node.getAttribute('href');
                                if (href && isScriptOrDataUrlLoose(href) && isScriptOrDataUrl(node.protocol)) {
                                    node.removeAttribute('href');
                                }
                            }
                            else if (isElementAction(node)) {
                                if (isScriptOrDataUrl(node.action)) {
                                    node.removeAttribute('action');
                                }
                            }
                            else if (isElementFormAction(node)) {
                                if (isScriptOrDataUrl(node.formAction)) {
                                    node.removeAttribute('formaction');
                                }
                            }
                            else if (isElementIframe(node)) {
                                if (isScriptOrDataUrl(node.src)) {
                                    node.removeAttribute('formaction');
                                }
                                node.setAttribute('sandbox', 'allow-scripts'); //TODO: This is kinda arbitrary, it should be customizable and more flexible
                            }
                        }
                    }
                }
            });
            return input;
        };
        this.sanitizeFor = (element, input) => {
            throw new Error('"sanitizeFor" is not implemented yet');
        };
        const { allowComments, allowCustomElements, allowUnknownMarkup, blockElements, dropElements, dropAttributes } = configuration;
        if (allowComments === false)
            throw new Error('A false "allowComments" is not supported yet');
        if (allowCustomElements)
            throw new Error('A true "allowCustomElements" is not supported yet');
        if (allowUnknownMarkup)
            throw new Error('A true "allowUnknownMarkup" is not supported yet');
        if (blockElements)
            throw new Error('"blockElements" is not supported yet, use "allowElements" instead');
        if (dropElements)
            throw new Error('"dropElements" is not supported yet, use "allowElements" instead');
        if (dropAttributes)
            throw new Error('"dropAttributes" is not supported yet, use "allowAttributes" instead');
        __classPrivateFieldSet(this, _Amuchina_configuration, cloneDeep(DEFAULTS), "f");
        const { allowElements, allowAttributes } = configuration;
        if (allowElements)
            __classPrivateFieldGet(this, _Amuchina_configuration, "f").allowElements = configuration.allowElements;
        if (allowAttributes)
            __classPrivateFieldGet(this, _Amuchina_configuration, "f").allowAttributes = configuration.allowAttributes;
        __classPrivateFieldSet(this, _Amuchina_allowElements, new Set(__classPrivateFieldGet(this, _Amuchina_configuration, "f").allowElements), "f");
        __classPrivateFieldSet(this, _Amuchina_allowAttributes, Object.fromEntries(Object.entries(__classPrivateFieldGet(this, _Amuchina_configuration, "f").allowAttributes || {}).map(([element, attributes]) => [element, new Set(attributes)])), "f");
    }
}
_Amuchina_configuration = new WeakMap(), _Amuchina_allowElements = new WeakMap(), _Amuchina_allowAttributes = new WeakMap();
/* STATIC API */
Amuchina.getDefaultConfiguration = () => {
    return cloneDeep(DEFAULTS);
};

export { Amuchina as A };
//# sourceMappingURL=index-C7e-J7CF.js.map
