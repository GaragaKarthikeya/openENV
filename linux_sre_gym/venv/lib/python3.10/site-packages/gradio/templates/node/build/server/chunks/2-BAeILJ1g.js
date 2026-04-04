import { b as browser, d as dev } from './index5-BZVOFaHm.js';
import { w as writable, d as derived } from './index-Cg-Pg6j3.js';
import './async-D55cHugf.js';

var commonjsGlobal = typeof globalThis !== "undefined" ? globalThis : typeof window !== "undefined" ? window : typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : {};
function getDefaultExportFromCjs(x) {
  return x && x.__esModule && Object.prototype.hasOwnProperty.call(x, "default") ? x["default"] : x;
}
var cjs;
var hasRequiredCjs;
function requireCjs() {
  if (hasRequiredCjs) return cjs;
  hasRequiredCjs = 1;
  var isMergeableObject = function isMergeableObject2(value) {
    return isNonNullObject(value) && !isSpecial(value);
  };
  function isNonNullObject(value) {
    return !!value && typeof value === "object";
  }
  function isSpecial(value) {
    var stringValue = Object.prototype.toString.call(value);
    return stringValue === "[object RegExp]" || stringValue === "[object Date]" || isReactElement(value);
  }
  var canUseSymbol = typeof Symbol === "function" && Symbol.for;
  var REACT_ELEMENT_TYPE = canUseSymbol ? /* @__PURE__ */ Symbol.for("react.element") : 60103;
  function isReactElement(value) {
    return value.$$typeof === REACT_ELEMENT_TYPE;
  }
  function emptyTarget(val) {
    return Array.isArray(val) ? [] : {};
  }
  function cloneUnlessOtherwiseSpecified(value, options2) {
    return options2.clone !== false && options2.isMergeableObject(value) ? deepmerge2(emptyTarget(value), value, options2) : value;
  }
  function defaultArrayMerge(target, source, options2) {
    return target.concat(source).map(function(element) {
      return cloneUnlessOtherwiseSpecified(element, options2);
    });
  }
  function getMergeFunction(key, options2) {
    if (!options2.customMerge) {
      return deepmerge2;
    }
    var customMerge = options2.customMerge(key);
    return typeof customMerge === "function" ? customMerge : deepmerge2;
  }
  function getEnumerableOwnPropertySymbols(target) {
    return Object.getOwnPropertySymbols ? Object.getOwnPropertySymbols(target).filter(function(symbol) {
      return Object.propertyIsEnumerable.call(target, symbol);
    }) : [];
  }
  function getKeys(target) {
    return Object.keys(target).concat(getEnumerableOwnPropertySymbols(target));
  }
  function propertyIsOnObject(object, property) {
    try {
      return property in object;
    } catch (_) {
      return false;
    }
  }
  function propertyIsUnsafe(target, key) {
    return propertyIsOnObject(target, key) && !(Object.hasOwnProperty.call(target, key) && Object.propertyIsEnumerable.call(target, key));
  }
  function mergeObject(target, source, options2) {
    var destination = {};
    if (options2.isMergeableObject(target)) {
      getKeys(target).forEach(function(key) {
        destination[key] = cloneUnlessOtherwiseSpecified(target[key], options2);
      });
    }
    getKeys(source).forEach(function(key) {
      if (propertyIsUnsafe(target, key)) {
        return;
      }
      if (propertyIsOnObject(target, key) && options2.isMergeableObject(source[key])) {
        destination[key] = getMergeFunction(key, options2)(target[key], source[key], options2);
      } else {
        destination[key] = cloneUnlessOtherwiseSpecified(source[key], options2);
      }
    });
    return destination;
  }
  function deepmerge2(target, source, options2) {
    options2 = options2 || {};
    options2.arrayMerge = options2.arrayMerge || defaultArrayMerge;
    options2.isMergeableObject = options2.isMergeableObject || isMergeableObject;
    options2.cloneUnlessOtherwiseSpecified = cloneUnlessOtherwiseSpecified;
    var sourceIsArray = Array.isArray(source);
    var targetIsArray = Array.isArray(target);
    var sourceAndTargetTypesMatch = sourceIsArray === targetIsArray;
    if (!sourceAndTargetTypesMatch) {
      return cloneUnlessOtherwiseSpecified(source, options2);
    } else if (sourceIsArray) {
      return options2.arrayMerge(target, source, options2);
    } else {
      return mergeObject(target, source, options2);
    }
  }
  deepmerge2.all = function deepmergeAll(array, options2) {
    if (!Array.isArray(array)) {
      throw new Error("first argument should be an array");
    }
    return array.reduce(function(prev, next) {
      return deepmerge2(prev, next, options2);
    }, {});
  };
  var deepmerge_1 = deepmerge2;
  cjs = deepmerge_1;
  return cjs;
}
var cjsExports = requireCjs();
const deepmerge = /* @__PURE__ */ getDefaultExportFromCjs(cjsExports);
var extendStatics = function(d, b) {
  extendStatics = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(d2, b2) {
    d2.__proto__ = b2;
  } || function(d2, b2) {
    for (var p in b2) if (Object.prototype.hasOwnProperty.call(b2, p)) d2[p] = b2[p];
  };
  return extendStatics(d, b);
};
function __extends(d, b) {
  if (typeof b !== "function" && b !== null)
    throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
  extendStatics(d, b);
  function __() {
    this.constructor = d;
  }
  d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
}
var __assign = function() {
  __assign = Object.assign || function __assign2(t) {
    for (var s, i = 1, n = arguments.length; i < n; i++) {
      s = arguments[i];
      for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p)) t[p] = s[p];
    }
    return t;
  };
  return __assign.apply(this, arguments);
};
function __rest(s, e) {
  var t = {};
  for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p) && e.indexOf(p) < 0)
    t[p] = s[p];
  if (s != null && typeof Object.getOwnPropertySymbols === "function")
    for (var i = 0, p = Object.getOwnPropertySymbols(s); i < p.length; i++) {
      if (e.indexOf(p[i]) < 0 && Object.prototype.propertyIsEnumerable.call(s, p[i]))
        t[p[i]] = s[p[i]];
    }
  return t;
}
function __spreadArray(to, from, pack) {
  if (arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
    if (ar || !(i in from)) {
      if (!ar) ar = Array.prototype.slice.call(from, 0, i);
      ar[i] = from[i];
    }
  }
  return to.concat(ar || Array.prototype.slice.call(from));
}
typeof SuppressedError === "function" ? SuppressedError : function(error, suppressed, message) {
  var e = new Error(message);
  return e.name = "SuppressedError", e.error = error, e.suppressed = suppressed, e;
};
function memoize(fn, options2) {
  var cache = options2 && options2.cache ? options2.cache : cacheDefault;
  var serializer = options2 && options2.serializer ? options2.serializer : serializerDefault;
  var strategy = options2 && options2.strategy ? options2.strategy : strategyDefault;
  return strategy(fn, {
    cache,
    serializer
  });
}
function isPrimitive(value) {
  return value == null || typeof value === "number" || typeof value === "boolean";
}
function monadic(fn, cache, serializer, arg) {
  var cacheKey = isPrimitive(arg) ? arg : serializer(arg);
  var computedValue = cache.get(cacheKey);
  if (typeof computedValue === "undefined") {
    computedValue = fn.call(this, arg);
    cache.set(cacheKey, computedValue);
  }
  return computedValue;
}
function variadic(fn, cache, serializer) {
  var args = Array.prototype.slice.call(arguments, 3);
  var cacheKey = serializer(args);
  var computedValue = cache.get(cacheKey);
  if (typeof computedValue === "undefined") {
    computedValue = fn.apply(this, args);
    cache.set(cacheKey, computedValue);
  }
  return computedValue;
}
function assemble(fn, context, strategy, cache, serialize) {
  return strategy.bind(context, fn, cache, serialize);
}
function strategyDefault(fn, options2) {
  var strategy = fn.length === 1 ? monadic : variadic;
  return assemble(fn, this, strategy, options2.cache.create(), options2.serializer);
}
function strategyVariadic(fn, options2) {
  return assemble(fn, this, variadic, options2.cache.create(), options2.serializer);
}
var serializerDefault = function() {
  return JSON.stringify(arguments);
};
var ObjectWithoutPrototypeCache = (
  /** @class */
  (function() {
    function ObjectWithoutPrototypeCache2() {
      this.cache = /* @__PURE__ */ Object.create(null);
    }
    ObjectWithoutPrototypeCache2.prototype.get = function(key) {
      return this.cache[key];
    };
    ObjectWithoutPrototypeCache2.prototype.set = function(key, value) {
      this.cache[key] = value;
    };
    return ObjectWithoutPrototypeCache2;
  })()
);
var cacheDefault = {
  create: function create() {
    return new ObjectWithoutPrototypeCache();
  }
};
var strategies = {
  variadic: strategyVariadic
};
var ErrorKind;
(function(ErrorKind2) {
  ErrorKind2[ErrorKind2["EXPECT_ARGUMENT_CLOSING_BRACE"] = 1] = "EXPECT_ARGUMENT_CLOSING_BRACE";
  ErrorKind2[ErrorKind2["EMPTY_ARGUMENT"] = 2] = "EMPTY_ARGUMENT";
  ErrorKind2[ErrorKind2["MALFORMED_ARGUMENT"] = 3] = "MALFORMED_ARGUMENT";
  ErrorKind2[ErrorKind2["EXPECT_ARGUMENT_TYPE"] = 4] = "EXPECT_ARGUMENT_TYPE";
  ErrorKind2[ErrorKind2["INVALID_ARGUMENT_TYPE"] = 5] = "INVALID_ARGUMENT_TYPE";
  ErrorKind2[ErrorKind2["EXPECT_ARGUMENT_STYLE"] = 6] = "EXPECT_ARGUMENT_STYLE";
  ErrorKind2[ErrorKind2["INVALID_NUMBER_SKELETON"] = 7] = "INVALID_NUMBER_SKELETON";
  ErrorKind2[ErrorKind2["INVALID_DATE_TIME_SKELETON"] = 8] = "INVALID_DATE_TIME_SKELETON";
  ErrorKind2[ErrorKind2["EXPECT_NUMBER_SKELETON"] = 9] = "EXPECT_NUMBER_SKELETON";
  ErrorKind2[ErrorKind2["EXPECT_DATE_TIME_SKELETON"] = 10] = "EXPECT_DATE_TIME_SKELETON";
  ErrorKind2[ErrorKind2["UNCLOSED_QUOTE_IN_ARGUMENT_STYLE"] = 11] = "UNCLOSED_QUOTE_IN_ARGUMENT_STYLE";
  ErrorKind2[ErrorKind2["EXPECT_SELECT_ARGUMENT_OPTIONS"] = 12] = "EXPECT_SELECT_ARGUMENT_OPTIONS";
  ErrorKind2[ErrorKind2["EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE"] = 13] = "EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE";
  ErrorKind2[ErrorKind2["INVALID_PLURAL_ARGUMENT_OFFSET_VALUE"] = 14] = "INVALID_PLURAL_ARGUMENT_OFFSET_VALUE";
  ErrorKind2[ErrorKind2["EXPECT_SELECT_ARGUMENT_SELECTOR"] = 15] = "EXPECT_SELECT_ARGUMENT_SELECTOR";
  ErrorKind2[ErrorKind2["EXPECT_PLURAL_ARGUMENT_SELECTOR"] = 16] = "EXPECT_PLURAL_ARGUMENT_SELECTOR";
  ErrorKind2[ErrorKind2["EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT"] = 17] = "EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT";
  ErrorKind2[ErrorKind2["EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT"] = 18] = "EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT";
  ErrorKind2[ErrorKind2["INVALID_PLURAL_ARGUMENT_SELECTOR"] = 19] = "INVALID_PLURAL_ARGUMENT_SELECTOR";
  ErrorKind2[ErrorKind2["DUPLICATE_PLURAL_ARGUMENT_SELECTOR"] = 20] = "DUPLICATE_PLURAL_ARGUMENT_SELECTOR";
  ErrorKind2[ErrorKind2["DUPLICATE_SELECT_ARGUMENT_SELECTOR"] = 21] = "DUPLICATE_SELECT_ARGUMENT_SELECTOR";
  ErrorKind2[ErrorKind2["MISSING_OTHER_CLAUSE"] = 22] = "MISSING_OTHER_CLAUSE";
  ErrorKind2[ErrorKind2["INVALID_TAG"] = 23] = "INVALID_TAG";
  ErrorKind2[ErrorKind2["INVALID_TAG_NAME"] = 25] = "INVALID_TAG_NAME";
  ErrorKind2[ErrorKind2["UNMATCHED_CLOSING_TAG"] = 26] = "UNMATCHED_CLOSING_TAG";
  ErrorKind2[ErrorKind2["UNCLOSED_TAG"] = 27] = "UNCLOSED_TAG";
})(ErrorKind || (ErrorKind = {}));
var TYPE;
(function(TYPE2) {
  TYPE2[TYPE2["literal"] = 0] = "literal";
  TYPE2[TYPE2["argument"] = 1] = "argument";
  TYPE2[TYPE2["number"] = 2] = "number";
  TYPE2[TYPE2["date"] = 3] = "date";
  TYPE2[TYPE2["time"] = 4] = "time";
  TYPE2[TYPE2["select"] = 5] = "select";
  TYPE2[TYPE2["plural"] = 6] = "plural";
  TYPE2[TYPE2["pound"] = 7] = "pound";
  TYPE2[TYPE2["tag"] = 8] = "tag";
})(TYPE || (TYPE = {}));
var SKELETON_TYPE;
(function(SKELETON_TYPE2) {
  SKELETON_TYPE2[SKELETON_TYPE2["number"] = 0] = "number";
  SKELETON_TYPE2[SKELETON_TYPE2["dateTime"] = 1] = "dateTime";
})(SKELETON_TYPE || (SKELETON_TYPE = {}));
function isLiteralElement(el) {
  return el.type === TYPE.literal;
}
function isArgumentElement(el) {
  return el.type === TYPE.argument;
}
function isNumberElement(el) {
  return el.type === TYPE.number;
}
function isDateElement(el) {
  return el.type === TYPE.date;
}
function isTimeElement(el) {
  return el.type === TYPE.time;
}
function isSelectElement(el) {
  return el.type === TYPE.select;
}
function isPluralElement(el) {
  return el.type === TYPE.plural;
}
function isPoundElement(el) {
  return el.type === TYPE.pound;
}
function isTagElement(el) {
  return el.type === TYPE.tag;
}
function isNumberSkeleton(el) {
  return !!(el && typeof el === "object" && el.type === SKELETON_TYPE.number);
}
function isDateTimeSkeleton(el) {
  return !!(el && typeof el === "object" && el.type === SKELETON_TYPE.dateTime);
}
var SPACE_SEPARATOR_REGEX = /[ \xA0\u1680\u2000-\u200A\u202F\u205F\u3000]/;
var DATE_TIME_REGEX = /(?:[Eec]{1,6}|G{1,5}|[Qq]{1,5}|(?:[yYur]+|U{1,5})|[ML]{1,5}|d{1,2}|D{1,3}|F{1}|[abB]{1,5}|[hkHK]{1,2}|w{1,2}|W{1}|m{1,2}|s{1,2}|[zZOvVxX]{1,4})(?=([^']*'[^']*')*[^']*$)/g;
function parseDateTimeSkeleton(skeleton) {
  var result = {};
  skeleton.replace(DATE_TIME_REGEX, function(match) {
    var len = match.length;
    switch (match[0]) {
      // Era
      case "G":
        result.era = len === 4 ? "long" : len === 5 ? "narrow" : "short";
        break;
      // Year
      case "y":
        result.year = len === 2 ? "2-digit" : "numeric";
        break;
      case "Y":
      case "u":
      case "U":
      case "r":
        throw new RangeError("`Y/u/U/r` (year) patterns are not supported, use `y` instead");
      // Quarter
      case "q":
      case "Q":
        throw new RangeError("`q/Q` (quarter) patterns are not supported");
      // Month
      case "M":
      case "L":
        result.month = ["numeric", "2-digit", "short", "long", "narrow"][len - 1];
        break;
      // Week
      case "w":
      case "W":
        throw new RangeError("`w/W` (week) patterns are not supported");
      case "d":
        result.day = ["numeric", "2-digit"][len - 1];
        break;
      case "D":
      case "F":
      case "g":
        throw new RangeError("`D/F/g` (day) patterns are not supported, use `d` instead");
      // Weekday
      case "E":
        result.weekday = len === 4 ? "long" : len === 5 ? "narrow" : "short";
        break;
      case "e":
        if (len < 4) {
          throw new RangeError("`e..eee` (weekday) patterns are not supported");
        }
        result.weekday = ["short", "long", "narrow", "short"][len - 4];
        break;
      case "c":
        if (len < 4) {
          throw new RangeError("`c..ccc` (weekday) patterns are not supported");
        }
        result.weekday = ["short", "long", "narrow", "short"][len - 4];
        break;
      // Period
      case "a":
        result.hour12 = true;
        break;
      case "b":
      // am, pm, noon, midnight
      case "B":
        throw new RangeError("`b/B` (period) patterns are not supported, use `a` instead");
      // Hour
      case "h":
        result.hourCycle = "h12";
        result.hour = ["numeric", "2-digit"][len - 1];
        break;
      case "H":
        result.hourCycle = "h23";
        result.hour = ["numeric", "2-digit"][len - 1];
        break;
      case "K":
        result.hourCycle = "h11";
        result.hour = ["numeric", "2-digit"][len - 1];
        break;
      case "k":
        result.hourCycle = "h24";
        result.hour = ["numeric", "2-digit"][len - 1];
        break;
      case "j":
      case "J":
      case "C":
        throw new RangeError("`j/J/C` (hour) patterns are not supported, use `h/H/K/k` instead");
      // Minute
      case "m":
        result.minute = ["numeric", "2-digit"][len - 1];
        break;
      // Second
      case "s":
        result.second = ["numeric", "2-digit"][len - 1];
        break;
      case "S":
      case "A":
        throw new RangeError("`S/A` (second) patterns are not supported, use `s` instead");
      // Zone
      case "z":
        result.timeZoneName = len < 4 ? "short" : "long";
        break;
      case "Z":
      // 1..3, 4, 5: The ISO8601 varios formats
      case "O":
      // 1, 4: milliseconds in day short, long
      case "v":
      // 1, 4: generic non-location format
      case "V":
      // 1, 2, 3, 4: time zone ID or city
      case "X":
      // 1, 2, 3, 4: The ISO8601 varios formats
      case "x":
        throw new RangeError("`Z/O/v/V/X/x` (timeZone) patterns are not supported, use `z` instead");
    }
    return "";
  });
  return result;
}
var WHITE_SPACE_REGEX = /[\t-\r \x85\u200E\u200F\u2028\u2029]/i;
function parseNumberSkeletonFromString(skeleton) {
  if (skeleton.length === 0) {
    throw new Error("Number skeleton cannot be empty");
  }
  var stringTokens = skeleton.split(WHITE_SPACE_REGEX).filter(function(x) {
    return x.length > 0;
  });
  var tokens = [];
  for (var _i = 0, stringTokens_1 = stringTokens; _i < stringTokens_1.length; _i++) {
    var stringToken = stringTokens_1[_i];
    var stemAndOptions = stringToken.split("/");
    if (stemAndOptions.length === 0) {
      throw new Error("Invalid number skeleton");
    }
    var stem = stemAndOptions[0], options2 = stemAndOptions.slice(1);
    for (var _a2 = 0, options_1 = options2; _a2 < options_1.length; _a2++) {
      var option = options_1[_a2];
      if (option.length === 0) {
        throw new Error("Invalid number skeleton");
      }
    }
    tokens.push({ stem, options: options2 });
  }
  return tokens;
}
function icuUnitToEcma(unit) {
  return unit.replace(/^(.*?)-/, "");
}
var FRACTION_PRECISION_REGEX = /^\.(?:(0+)(\*)?|(#+)|(0+)(#+))$/g;
var SIGNIFICANT_PRECISION_REGEX = /^(@+)?(\+|#+)?[rs]?$/g;
var INTEGER_WIDTH_REGEX = /(\*)(0+)|(#+)(0+)|(0+)/g;
var CONCISE_INTEGER_WIDTH_REGEX = /^(0+)$/;
function parseSignificantPrecision(str) {
  var result = {};
  if (str[str.length - 1] === "r") {
    result.roundingPriority = "morePrecision";
  } else if (str[str.length - 1] === "s") {
    result.roundingPriority = "lessPrecision";
  }
  str.replace(SIGNIFICANT_PRECISION_REGEX, function(_, g1, g2) {
    if (typeof g2 !== "string") {
      result.minimumSignificantDigits = g1.length;
      result.maximumSignificantDigits = g1.length;
    } else if (g2 === "+") {
      result.minimumSignificantDigits = g1.length;
    } else if (g1[0] === "#") {
      result.maximumSignificantDigits = g1.length;
    } else {
      result.minimumSignificantDigits = g1.length;
      result.maximumSignificantDigits = g1.length + (typeof g2 === "string" ? g2.length : 0);
    }
    return "";
  });
  return result;
}
function parseSign(str) {
  switch (str) {
    case "sign-auto":
      return {
        signDisplay: "auto"
      };
    case "sign-accounting":
    case "()":
      return {
        currencySign: "accounting"
      };
    case "sign-always":
    case "+!":
      return {
        signDisplay: "always"
      };
    case "sign-accounting-always":
    case "()!":
      return {
        signDisplay: "always",
        currencySign: "accounting"
      };
    case "sign-except-zero":
    case "+?":
      return {
        signDisplay: "exceptZero"
      };
    case "sign-accounting-except-zero":
    case "()?":
      return {
        signDisplay: "exceptZero",
        currencySign: "accounting"
      };
    case "sign-never":
    case "+_":
      return {
        signDisplay: "never"
      };
  }
}
function parseConciseScientificAndEngineeringStem(stem) {
  var result;
  if (stem[0] === "E" && stem[1] === "E") {
    result = {
      notation: "engineering"
    };
    stem = stem.slice(2);
  } else if (stem[0] === "E") {
    result = {
      notation: "scientific"
    };
    stem = stem.slice(1);
  }
  if (result) {
    var signDisplay = stem.slice(0, 2);
    if (signDisplay === "+!") {
      result.signDisplay = "always";
      stem = stem.slice(2);
    } else if (signDisplay === "+?") {
      result.signDisplay = "exceptZero";
      stem = stem.slice(2);
    }
    if (!CONCISE_INTEGER_WIDTH_REGEX.test(stem)) {
      throw new Error("Malformed concise eng/scientific notation");
    }
    result.minimumIntegerDigits = stem.length;
  }
  return result;
}
function parseNotationOptions(opt) {
  var result = {};
  var signOpts = parseSign(opt);
  if (signOpts) {
    return signOpts;
  }
  return result;
}
function parseNumberSkeleton(tokens) {
  var result = {};
  for (var _i = 0, tokens_1 = tokens; _i < tokens_1.length; _i++) {
    var token = tokens_1[_i];
    switch (token.stem) {
      case "percent":
      case "%":
        result.style = "percent";
        continue;
      case "%x100":
        result.style = "percent";
        result.scale = 100;
        continue;
      case "currency":
        result.style = "currency";
        result.currency = token.options[0];
        continue;
      case "group-off":
      case ",_":
        result.useGrouping = false;
        continue;
      case "precision-integer":
      case ".":
        result.maximumFractionDigits = 0;
        continue;
      case "measure-unit":
      case "unit":
        result.style = "unit";
        result.unit = icuUnitToEcma(token.options[0]);
        continue;
      case "compact-short":
      case "K":
        result.notation = "compact";
        result.compactDisplay = "short";
        continue;
      case "compact-long":
      case "KK":
        result.notation = "compact";
        result.compactDisplay = "long";
        continue;
      case "scientific":
        result = __assign(__assign(__assign({}, result), { notation: "scientific" }), token.options.reduce(function(all, opt2) {
          return __assign(__assign({}, all), parseNotationOptions(opt2));
        }, {}));
        continue;
      case "engineering":
        result = __assign(__assign(__assign({}, result), { notation: "engineering" }), token.options.reduce(function(all, opt2) {
          return __assign(__assign({}, all), parseNotationOptions(opt2));
        }, {}));
        continue;
      case "notation-simple":
        result.notation = "standard";
        continue;
      // https://github.com/unicode-org/icu/blob/master/icu4c/source/i18n/unicode/unumberformatter.h
      case "unit-width-narrow":
        result.currencyDisplay = "narrowSymbol";
        result.unitDisplay = "narrow";
        continue;
      case "unit-width-short":
        result.currencyDisplay = "code";
        result.unitDisplay = "short";
        continue;
      case "unit-width-full-name":
        result.currencyDisplay = "name";
        result.unitDisplay = "long";
        continue;
      case "unit-width-iso-code":
        result.currencyDisplay = "symbol";
        continue;
      case "scale":
        result.scale = parseFloat(token.options[0]);
        continue;
      case "rounding-mode-floor":
        result.roundingMode = "floor";
        continue;
      case "rounding-mode-ceiling":
        result.roundingMode = "ceil";
        continue;
      case "rounding-mode-down":
        result.roundingMode = "trunc";
        continue;
      case "rounding-mode-up":
        result.roundingMode = "expand";
        continue;
      case "rounding-mode-half-even":
        result.roundingMode = "halfEven";
        continue;
      case "rounding-mode-half-down":
        result.roundingMode = "halfTrunc";
        continue;
      case "rounding-mode-half-up":
        result.roundingMode = "halfExpand";
        continue;
      // https://unicode-org.github.io/icu/userguide/format_parse/numbers/skeletons.html#integer-width
      case "integer-width":
        if (token.options.length > 1) {
          throw new RangeError("integer-width stems only accept a single optional option");
        }
        token.options[0].replace(INTEGER_WIDTH_REGEX, function(_, g1, g2, g3, g4, g5) {
          if (g1) {
            result.minimumIntegerDigits = g2.length;
          } else if (g3 && g4) {
            throw new Error("We currently do not support maximum integer digits");
          } else if (g5) {
            throw new Error("We currently do not support exact integer digits");
          }
          return "";
        });
        continue;
    }
    if (CONCISE_INTEGER_WIDTH_REGEX.test(token.stem)) {
      result.minimumIntegerDigits = token.stem.length;
      continue;
    }
    if (FRACTION_PRECISION_REGEX.test(token.stem)) {
      if (token.options.length > 1) {
        throw new RangeError("Fraction-precision stems only accept a single optional option");
      }
      token.stem.replace(FRACTION_PRECISION_REGEX, function(_, g1, g2, g3, g4, g5) {
        if (g2 === "*") {
          result.minimumFractionDigits = g1.length;
        } else if (g3 && g3[0] === "#") {
          result.maximumFractionDigits = g3.length;
        } else if (g4 && g5) {
          result.minimumFractionDigits = g4.length;
          result.maximumFractionDigits = g4.length + g5.length;
        } else {
          result.minimumFractionDigits = g1.length;
          result.maximumFractionDigits = g1.length;
        }
        return "";
      });
      var opt = token.options[0];
      if (opt === "w") {
        result = __assign(__assign({}, result), { trailingZeroDisplay: "stripIfInteger" });
      } else if (opt) {
        result = __assign(__assign({}, result), parseSignificantPrecision(opt));
      }
      continue;
    }
    if (SIGNIFICANT_PRECISION_REGEX.test(token.stem)) {
      result = __assign(__assign({}, result), parseSignificantPrecision(token.stem));
      continue;
    }
    var signOpts = parseSign(token.stem);
    if (signOpts) {
      result = __assign(__assign({}, result), signOpts);
    }
    var conciseScientificAndEngineeringOpts = parseConciseScientificAndEngineeringStem(token.stem);
    if (conciseScientificAndEngineeringOpts) {
      result = __assign(__assign({}, result), conciseScientificAndEngineeringOpts);
    }
  }
  return result;
}
var timeData = {
  "001": [
    "H",
    "h"
  ],
  "419": [
    "h",
    "H",
    "hB",
    "hb"
  ],
  "AC": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "AD": [
    "H",
    "hB"
  ],
  "AE": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "AF": [
    "H",
    "hb",
    "hB",
    "h"
  ],
  "AG": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "AI": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "AL": [
    "h",
    "H",
    "hB"
  ],
  "AM": [
    "H",
    "hB"
  ],
  "AO": [
    "H",
    "hB"
  ],
  "AR": [
    "h",
    "H",
    "hB",
    "hb"
  ],
  "AS": [
    "h",
    "H"
  ],
  "AT": [
    "H",
    "hB"
  ],
  "AU": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "AW": [
    "H",
    "hB"
  ],
  "AX": [
    "H"
  ],
  "AZ": [
    "H",
    "hB",
    "h"
  ],
  "BA": [
    "H",
    "hB",
    "h"
  ],
  "BB": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "BD": [
    "h",
    "hB",
    "H"
  ],
  "BE": [
    "H",
    "hB"
  ],
  "BF": [
    "H",
    "hB"
  ],
  "BG": [
    "H",
    "hB",
    "h"
  ],
  "BH": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "BI": [
    "H",
    "h"
  ],
  "BJ": [
    "H",
    "hB"
  ],
  "BL": [
    "H",
    "hB"
  ],
  "BM": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "BN": [
    "hb",
    "hB",
    "h",
    "H"
  ],
  "BO": [
    "h",
    "H",
    "hB",
    "hb"
  ],
  "BQ": [
    "H"
  ],
  "BR": [
    "H",
    "hB"
  ],
  "BS": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "BT": [
    "h",
    "H"
  ],
  "BW": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "BY": [
    "H",
    "h"
  ],
  "BZ": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "CA": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "CC": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "CD": [
    "hB",
    "H"
  ],
  "CF": [
    "H",
    "h",
    "hB"
  ],
  "CG": [
    "H",
    "hB"
  ],
  "CH": [
    "H",
    "hB",
    "h"
  ],
  "CI": [
    "H",
    "hB"
  ],
  "CK": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "CL": [
    "h",
    "H",
    "hB",
    "hb"
  ],
  "CM": [
    "H",
    "h",
    "hB"
  ],
  "CN": [
    "H",
    "hB",
    "hb",
    "h"
  ],
  "CO": [
    "h",
    "H",
    "hB",
    "hb"
  ],
  "CP": [
    "H"
  ],
  "CR": [
    "h",
    "H",
    "hB",
    "hb"
  ],
  "CU": [
    "h",
    "H",
    "hB",
    "hb"
  ],
  "CV": [
    "H",
    "hB"
  ],
  "CW": [
    "H",
    "hB"
  ],
  "CX": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "CY": [
    "h",
    "H",
    "hb",
    "hB"
  ],
  "CZ": [
    "H"
  ],
  "DE": [
    "H",
    "hB"
  ],
  "DG": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "DJ": [
    "h",
    "H"
  ],
  "DK": [
    "H"
  ],
  "DM": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "DO": [
    "h",
    "H",
    "hB",
    "hb"
  ],
  "DZ": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "EA": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  "EC": [
    "h",
    "H",
    "hB",
    "hb"
  ],
  "EE": [
    "H",
    "hB"
  ],
  "EG": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "EH": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "ER": [
    "h",
    "H"
  ],
  "ES": [
    "H",
    "hB",
    "h",
    "hb"
  ],
  "ET": [
    "hB",
    "hb",
    "h",
    "H"
  ],
  "FI": [
    "H"
  ],
  "FJ": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "FK": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "FM": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "FO": [
    "H",
    "h"
  ],
  "FR": [
    "H",
    "hB"
  ],
  "GA": [
    "H",
    "hB"
  ],
  "GB": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "GD": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "GE": [
    "H",
    "hB",
    "h"
  ],
  "GF": [
    "H",
    "hB"
  ],
  "GG": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "GH": [
    "h",
    "H"
  ],
  "GI": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "GL": [
    "H",
    "h"
  ],
  "GM": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "GN": [
    "H",
    "hB"
  ],
  "GP": [
    "H",
    "hB"
  ],
  "GQ": [
    "H",
    "hB",
    "h",
    "hb"
  ],
  "GR": [
    "h",
    "H",
    "hb",
    "hB"
  ],
  "GT": [
    "h",
    "H",
    "hB",
    "hb"
  ],
  "GU": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "GW": [
    "H",
    "hB"
  ],
  "GY": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "HK": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "HN": [
    "h",
    "H",
    "hB",
    "hb"
  ],
  "HR": [
    "H",
    "hB"
  ],
  "HU": [
    "H",
    "h"
  ],
  "IC": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  "ID": [
    "H"
  ],
  "IE": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "IL": [
    "H",
    "hB"
  ],
  "IM": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "IN": [
    "h",
    "H"
  ],
  "IO": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "IQ": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "IR": [
    "hB",
    "H"
  ],
  "IS": [
    "H"
  ],
  "IT": [
    "H",
    "hB"
  ],
  "JE": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "JM": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "JO": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "JP": [
    "H",
    "K",
    "h"
  ],
  "KE": [
    "hB",
    "hb",
    "H",
    "h"
  ],
  "KG": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  "KH": [
    "hB",
    "h",
    "H",
    "hb"
  ],
  "KI": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "KM": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  "KN": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "KP": [
    "h",
    "H",
    "hB",
    "hb"
  ],
  "KR": [
    "h",
    "H",
    "hB",
    "hb"
  ],
  "KW": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "KY": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "KZ": [
    "H",
    "hB"
  ],
  "LA": [
    "H",
    "hb",
    "hB",
    "h"
  ],
  "LB": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "LC": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "LI": [
    "H",
    "hB",
    "h"
  ],
  "LK": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  "LR": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "LS": [
    "h",
    "H"
  ],
  "LT": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "LU": [
    "H",
    "h",
    "hB"
  ],
  "LV": [
    "H",
    "hB",
    "hb",
    "h"
  ],
  "LY": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "MA": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  "MC": [
    "H",
    "hB"
  ],
  "MD": [
    "H",
    "hB"
  ],
  "ME": [
    "H",
    "hB",
    "h"
  ],
  "MF": [
    "H",
    "hB"
  ],
  "MG": [
    "H",
    "h"
  ],
  "MH": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "MK": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "ML": [
    "H"
  ],
  "MM": [
    "hB",
    "hb",
    "H",
    "h"
  ],
  "MN": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "MO": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "MP": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "MQ": [
    "H",
    "hB"
  ],
  "MR": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "MS": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "MT": [
    "H",
    "h"
  ],
  "MU": [
    "H",
    "h"
  ],
  "MV": [
    "H",
    "h"
  ],
  "MW": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "MX": [
    "h",
    "H",
    "hB",
    "hb"
  ],
  "MY": [
    "hb",
    "hB",
    "h",
    "H"
  ],
  "MZ": [
    "H",
    "hB"
  ],
  "NA": [
    "h",
    "H",
    "hB",
    "hb"
  ],
  "NC": [
    "H",
    "hB"
  ],
  "NE": [
    "H"
  ],
  "NF": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "NG": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "NI": [
    "h",
    "H",
    "hB",
    "hb"
  ],
  "NL": [
    "H",
    "hB"
  ],
  "NO": [
    "H",
    "h"
  ],
  "NP": [
    "H",
    "h",
    "hB"
  ],
  "NR": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "NU": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "NZ": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "OM": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "PA": [
    "h",
    "H",
    "hB",
    "hb"
  ],
  "PE": [
    "h",
    "H",
    "hB",
    "hb"
  ],
  "PF": [
    "H",
    "h",
    "hB"
  ],
  "PG": [
    "h",
    "H"
  ],
  "PH": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "PK": [
    "h",
    "hB",
    "H"
  ],
  "PL": [
    "H",
    "h"
  ],
  "PM": [
    "H",
    "hB"
  ],
  "PN": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "PR": [
    "h",
    "H",
    "hB",
    "hb"
  ],
  "PS": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "PT": [
    "H",
    "hB"
  ],
  "PW": [
    "h",
    "H"
  ],
  "PY": [
    "h",
    "H",
    "hB",
    "hb"
  ],
  "QA": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "RE": [
    "H",
    "hB"
  ],
  "RO": [
    "H",
    "hB"
  ],
  "RS": [
    "H",
    "hB",
    "h"
  ],
  "RU": [
    "H"
  ],
  "RW": [
    "H",
    "h"
  ],
  "SA": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "SB": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "SC": [
    "H",
    "h",
    "hB"
  ],
  "SD": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "SE": [
    "H"
  ],
  "SG": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "SH": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "SI": [
    "H",
    "hB"
  ],
  "SJ": [
    "H"
  ],
  "SK": [
    "H"
  ],
  "SL": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "SM": [
    "H",
    "h",
    "hB"
  ],
  "SN": [
    "H",
    "h",
    "hB"
  ],
  "SO": [
    "h",
    "H"
  ],
  "SR": [
    "H",
    "hB"
  ],
  "SS": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "ST": [
    "H",
    "hB"
  ],
  "SV": [
    "h",
    "H",
    "hB",
    "hb"
  ],
  "SX": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "SY": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "SZ": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "TA": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "TC": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "TD": [
    "h",
    "H",
    "hB"
  ],
  "TF": [
    "H",
    "h",
    "hB"
  ],
  "TG": [
    "H",
    "hB"
  ],
  "TH": [
    "H",
    "h"
  ],
  "TJ": [
    "H",
    "h"
  ],
  "TL": [
    "H",
    "hB",
    "hb",
    "h"
  ],
  "TM": [
    "H",
    "h"
  ],
  "TN": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "TO": [
    "h",
    "H"
  ],
  "TR": [
    "H",
    "hB"
  ],
  "TT": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "TW": [
    "hB",
    "hb",
    "h",
    "H"
  ],
  "TZ": [
    "hB",
    "hb",
    "H",
    "h"
  ],
  "UA": [
    "H",
    "hB",
    "h"
  ],
  "UG": [
    "hB",
    "hb",
    "H",
    "h"
  ],
  "UM": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "US": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "UY": [
    "h",
    "H",
    "hB",
    "hb"
  ],
  "UZ": [
    "H",
    "hB",
    "h"
  ],
  "VA": [
    "H",
    "h",
    "hB"
  ],
  "VC": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "VE": [
    "h",
    "H",
    "hB",
    "hb"
  ],
  "VG": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "VI": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "VN": [
    "H",
    "h"
  ],
  "VU": [
    "h",
    "H"
  ],
  "WF": [
    "H",
    "hB"
  ],
  "WS": [
    "h",
    "H"
  ],
  "XK": [
    "H",
    "hB",
    "h"
  ],
  "YE": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "YT": [
    "H",
    "hB"
  ],
  "ZA": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "ZM": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "ZW": [
    "H",
    "h"
  ],
  "af-ZA": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  "ar-001": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "ca-ES": [
    "H",
    "h",
    "hB"
  ],
  "en-001": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "en-HK": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "en-IL": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "en-MY": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "es-BR": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  "es-ES": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  "es-GQ": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  "fr-CA": [
    "H",
    "h",
    "hB"
  ],
  "gl-ES": [
    "H",
    "h",
    "hB"
  ],
  "gu-IN": [
    "hB",
    "hb",
    "h",
    "H"
  ],
  "hi-IN": [
    "hB",
    "h",
    "H"
  ],
  "it-CH": [
    "H",
    "h",
    "hB"
  ],
  "it-IT": [
    "H",
    "h",
    "hB"
  ],
  "kn-IN": [
    "hB",
    "h",
    "H"
  ],
  "ml-IN": [
    "hB",
    "h",
    "H"
  ],
  "mr-IN": [
    "hB",
    "hb",
    "h",
    "H"
  ],
  "pa-IN": [
    "hB",
    "hb",
    "h",
    "H"
  ],
  "ta-IN": [
    "hB",
    "h",
    "hb",
    "H"
  ],
  "te-IN": [
    "hB",
    "h",
    "H"
  ],
  "zu-ZA": [
    "H",
    "hB",
    "hb",
    "h"
  ]
};
function getBestPattern(skeleton, locale) {
  var skeletonCopy = "";
  for (var patternPos = 0; patternPos < skeleton.length; patternPos++) {
    var patternChar = skeleton.charAt(patternPos);
    if (patternChar === "j") {
      var extraLength = 0;
      while (patternPos + 1 < skeleton.length && skeleton.charAt(patternPos + 1) === patternChar) {
        extraLength++;
        patternPos++;
      }
      var hourLen = 1 + (extraLength & 1);
      var dayPeriodLen = extraLength < 2 ? 1 : 3 + (extraLength >> 1);
      var dayPeriodChar = "a";
      var hourChar = getDefaultHourSymbolFromLocale(locale);
      if (hourChar == "H" || hourChar == "k") {
        dayPeriodLen = 0;
      }
      while (dayPeriodLen-- > 0) {
        skeletonCopy += dayPeriodChar;
      }
      while (hourLen-- > 0) {
        skeletonCopy = hourChar + skeletonCopy;
      }
    } else if (patternChar === "J") {
      skeletonCopy += "H";
    } else {
      skeletonCopy += patternChar;
    }
  }
  return skeletonCopy;
}
function getDefaultHourSymbolFromLocale(locale) {
  var hourCycle = locale.hourCycle;
  if (hourCycle === void 0 && // @ts-ignore hourCycle(s) is not identified yet
  locale.hourCycles && // @ts-ignore
  locale.hourCycles.length) {
    hourCycle = locale.hourCycles[0];
  }
  if (hourCycle) {
    switch (hourCycle) {
      case "h24":
        return "k";
      case "h23":
        return "H";
      case "h12":
        return "h";
      case "h11":
        return "K";
      default:
        throw new Error("Invalid hourCycle");
    }
  }
  var languageTag = locale.language;
  var regionTag;
  if (languageTag !== "root") {
    regionTag = locale.maximize().region;
  }
  var hourCycles = timeData[regionTag || ""] || timeData[languageTag || ""] || timeData["".concat(languageTag, "-001")] || timeData["001"];
  return hourCycles[0];
}
var _a;
var SPACE_SEPARATOR_START_REGEX = new RegExp("^".concat(SPACE_SEPARATOR_REGEX.source, "*"));
var SPACE_SEPARATOR_END_REGEX = new RegExp("".concat(SPACE_SEPARATOR_REGEX.source, "*$"));
function createLocation(start, end) {
  return { start, end };
}
var hasNativeStartsWith = !!String.prototype.startsWith && "_a".startsWith("a", 1);
var hasNativeFromCodePoint = !!String.fromCodePoint;
var hasNativeFromEntries = !!Object.fromEntries;
var hasNativeCodePointAt = !!String.prototype.codePointAt;
var hasTrimStart = !!String.prototype.trimStart;
var hasTrimEnd = !!String.prototype.trimEnd;
var hasNativeIsSafeInteger = !!Number.isSafeInteger;
var isSafeInteger = hasNativeIsSafeInteger ? Number.isSafeInteger : function(n) {
  return typeof n === "number" && isFinite(n) && Math.floor(n) === n && Math.abs(n) <= 9007199254740991;
};
var REGEX_SUPPORTS_U_AND_Y = true;
try {
  var re = RE("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  REGEX_SUPPORTS_U_AND_Y = ((_a = re.exec("a")) === null || _a === void 0 ? void 0 : _a[0]) === "a";
} catch (_) {
  REGEX_SUPPORTS_U_AND_Y = false;
}
var startsWith = hasNativeStartsWith ? (
  // Native
  function startsWith2(s, search, position) {
    return s.startsWith(search, position);
  }
) : (
  // For IE11
  function startsWith3(s, search, position) {
    return s.slice(position, position + search.length) === search;
  }
);
var fromCodePoint = hasNativeFromCodePoint ? String.fromCodePoint : (
  // IE11
  function fromCodePoint2() {
    var codePoints = [];
    for (var _i = 0; _i < arguments.length; _i++) {
      codePoints[_i] = arguments[_i];
    }
    var elements = "";
    var length = codePoints.length;
    var i = 0;
    var code2;
    while (length > i) {
      code2 = codePoints[i++];
      if (code2 > 1114111)
        throw RangeError(code2 + " is not a valid code point");
      elements += code2 < 65536 ? String.fromCharCode(code2) : String.fromCharCode(((code2 -= 65536) >> 10) + 55296, code2 % 1024 + 56320);
    }
    return elements;
  }
);
var fromEntries = (
  // native
  hasNativeFromEntries ? Object.fromEntries : (
    // Ponyfill
    function fromEntries2(entries) {
      var obj = {};
      for (var _i = 0, entries_1 = entries; _i < entries_1.length; _i++) {
        var _a2 = entries_1[_i], k = _a2[0], v = _a2[1];
        obj[k] = v;
      }
      return obj;
    }
  )
);
var codePointAt = hasNativeCodePointAt ? (
  // Native
  function codePointAt2(s, index) {
    return s.codePointAt(index);
  }
) : (
  // IE 11
  function codePointAt3(s, index) {
    var size = s.length;
    if (index < 0 || index >= size) {
      return void 0;
    }
    var first = s.charCodeAt(index);
    var second;
    return first < 55296 || first > 56319 || index + 1 === size || (second = s.charCodeAt(index + 1)) < 56320 || second > 57343 ? first : (first - 55296 << 10) + (second - 56320) + 65536;
  }
);
var trimStart = hasTrimStart ? (
  // Native
  function trimStart2(s) {
    return s.trimStart();
  }
) : (
  // Ponyfill
  function trimStart3(s) {
    return s.replace(SPACE_SEPARATOR_START_REGEX, "");
  }
);
var trimEnd = hasTrimEnd ? (
  // Native
  function trimEnd2(s) {
    return s.trimEnd();
  }
) : (
  // Ponyfill
  function trimEnd3(s) {
    return s.replace(SPACE_SEPARATOR_END_REGEX, "");
  }
);
function RE(s, flag) {
  return new RegExp(s, flag);
}
var matchIdentifierAtIndex;
if (REGEX_SUPPORTS_U_AND_Y) {
  var IDENTIFIER_PREFIX_RE_1 = RE("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  matchIdentifierAtIndex = function matchIdentifierAtIndex2(s, index) {
    var _a2;
    IDENTIFIER_PREFIX_RE_1.lastIndex = index;
    var match = IDENTIFIER_PREFIX_RE_1.exec(s);
    return (_a2 = match[1]) !== null && _a2 !== void 0 ? _a2 : "";
  };
} else {
  matchIdentifierAtIndex = function matchIdentifierAtIndex2(s, index) {
    var match = [];
    while (true) {
      var c = codePointAt(s, index);
      if (c === void 0 || _isWhiteSpace(c) || _isPatternSyntax(c)) {
        break;
      }
      match.push(c);
      index += c >= 65536 ? 2 : 1;
    }
    return fromCodePoint.apply(void 0, match);
  };
}
var Parser = (
  /** @class */
  (function() {
    function Parser2(message, options2) {
      if (options2 === void 0) {
        options2 = {};
      }
      this.message = message;
      this.position = { offset: 0, line: 1, column: 1 };
      this.ignoreTag = !!options2.ignoreTag;
      this.locale = options2.locale;
      this.requiresOtherClause = !!options2.requiresOtherClause;
      this.shouldParseSkeletons = !!options2.shouldParseSkeletons;
    }
    Parser2.prototype.parse = function() {
      if (this.offset() !== 0) {
        throw Error("parser can only be used once");
      }
      return this.parseMessage(0, "", false);
    };
    Parser2.prototype.parseMessage = function(nestingLevel, parentArgType, expectingCloseTag) {
      var elements = [];
      while (!this.isEOF()) {
        var char = this.char();
        if (char === 123) {
          var result = this.parseArgument(nestingLevel, expectingCloseTag);
          if (result.err) {
            return result;
          }
          elements.push(result.val);
        } else if (char === 125 && nestingLevel > 0) {
          break;
        } else if (char === 35 && (parentArgType === "plural" || parentArgType === "selectordinal")) {
          var position = this.clonePosition();
          this.bump();
          elements.push({
            type: TYPE.pound,
            location: createLocation(position, this.clonePosition())
          });
        } else if (char === 60 && !this.ignoreTag && this.peek() === 47) {
          if (expectingCloseTag) {
            break;
          } else {
            return this.error(ErrorKind.UNMATCHED_CLOSING_TAG, createLocation(this.clonePosition(), this.clonePosition()));
          }
        } else if (char === 60 && !this.ignoreTag && _isAlpha(this.peek() || 0)) {
          var result = this.parseTag(nestingLevel, parentArgType);
          if (result.err) {
            return result;
          }
          elements.push(result.val);
        } else {
          var result = this.parseLiteral(nestingLevel, parentArgType);
          if (result.err) {
            return result;
          }
          elements.push(result.val);
        }
      }
      return { val: elements, err: null };
    };
    Parser2.prototype.parseTag = function(nestingLevel, parentArgType) {
      var startPosition = this.clonePosition();
      this.bump();
      var tagName = this.parseTagName();
      this.bumpSpace();
      if (this.bumpIf("/>")) {
        return {
          val: {
            type: TYPE.literal,
            value: "<".concat(tagName, "/>"),
            location: createLocation(startPosition, this.clonePosition())
          },
          err: null
        };
      } else if (this.bumpIf(">")) {
        var childrenResult = this.parseMessage(nestingLevel + 1, parentArgType, true);
        if (childrenResult.err) {
          return childrenResult;
        }
        var children = childrenResult.val;
        var endTagStartPosition = this.clonePosition();
        if (this.bumpIf("</")) {
          if (this.isEOF() || !_isAlpha(this.char())) {
            return this.error(ErrorKind.INVALID_TAG, createLocation(endTagStartPosition, this.clonePosition()));
          }
          var closingTagNameStartPosition = this.clonePosition();
          var closingTagName = this.parseTagName();
          if (tagName !== closingTagName) {
            return this.error(ErrorKind.UNMATCHED_CLOSING_TAG, createLocation(closingTagNameStartPosition, this.clonePosition()));
          }
          this.bumpSpace();
          if (!this.bumpIf(">")) {
            return this.error(ErrorKind.INVALID_TAG, createLocation(endTagStartPosition, this.clonePosition()));
          }
          return {
            val: {
              type: TYPE.tag,
              value: tagName,
              children,
              location: createLocation(startPosition, this.clonePosition())
            },
            err: null
          };
        } else {
          return this.error(ErrorKind.UNCLOSED_TAG, createLocation(startPosition, this.clonePosition()));
        }
      } else {
        return this.error(ErrorKind.INVALID_TAG, createLocation(startPosition, this.clonePosition()));
      }
    };
    Parser2.prototype.parseTagName = function() {
      var startOffset = this.offset();
      this.bump();
      while (!this.isEOF() && _isPotentialElementNameChar(this.char())) {
        this.bump();
      }
      return this.message.slice(startOffset, this.offset());
    };
    Parser2.prototype.parseLiteral = function(nestingLevel, parentArgType) {
      var start = this.clonePosition();
      var value = "";
      while (true) {
        var parseQuoteResult = this.tryParseQuote(parentArgType);
        if (parseQuoteResult) {
          value += parseQuoteResult;
          continue;
        }
        var parseUnquotedResult = this.tryParseUnquoted(nestingLevel, parentArgType);
        if (parseUnquotedResult) {
          value += parseUnquotedResult;
          continue;
        }
        var parseLeftAngleResult = this.tryParseLeftAngleBracket();
        if (parseLeftAngleResult) {
          value += parseLeftAngleResult;
          continue;
        }
        break;
      }
      var location2 = createLocation(start, this.clonePosition());
      return {
        val: { type: TYPE.literal, value, location: location2 },
        err: null
      };
    };
    Parser2.prototype.tryParseLeftAngleBracket = function() {
      if (!this.isEOF() && this.char() === 60 && (this.ignoreTag || // If at the opening tag or closing tag position, bail.
      !_isAlphaOrSlash(this.peek() || 0))) {
        this.bump();
        return "<";
      }
      return null;
    };
    Parser2.prototype.tryParseQuote = function(parentArgType) {
      if (this.isEOF() || this.char() !== 39) {
        return null;
      }
      switch (this.peek()) {
        case 39:
          this.bump();
          this.bump();
          return "'";
        // '{', '<', '>', '}'
        case 123:
        case 60:
        case 62:
        case 125:
          break;
        case 35:
          if (parentArgType === "plural" || parentArgType === "selectordinal") {
            break;
          }
          return null;
        default:
          return null;
      }
      this.bump();
      var codePoints = [this.char()];
      this.bump();
      while (!this.isEOF()) {
        var ch = this.char();
        if (ch === 39) {
          if (this.peek() === 39) {
            codePoints.push(39);
            this.bump();
          } else {
            this.bump();
            break;
          }
        } else {
          codePoints.push(ch);
        }
        this.bump();
      }
      return fromCodePoint.apply(void 0, codePoints);
    };
    Parser2.prototype.tryParseUnquoted = function(nestingLevel, parentArgType) {
      if (this.isEOF()) {
        return null;
      }
      var ch = this.char();
      if (ch === 60 || ch === 123 || ch === 35 && (parentArgType === "plural" || parentArgType === "selectordinal") || ch === 125 && nestingLevel > 0) {
        return null;
      } else {
        this.bump();
        return fromCodePoint(ch);
      }
    };
    Parser2.prototype.parseArgument = function(nestingLevel, expectingCloseTag) {
      var openingBracePosition = this.clonePosition();
      this.bump();
      this.bumpSpace();
      if (this.isEOF()) {
        return this.error(ErrorKind.EXPECT_ARGUMENT_CLOSING_BRACE, createLocation(openingBracePosition, this.clonePosition()));
      }
      if (this.char() === 125) {
        this.bump();
        return this.error(ErrorKind.EMPTY_ARGUMENT, createLocation(openingBracePosition, this.clonePosition()));
      }
      var value = this.parseIdentifierIfPossible().value;
      if (!value) {
        return this.error(ErrorKind.MALFORMED_ARGUMENT, createLocation(openingBracePosition, this.clonePosition()));
      }
      this.bumpSpace();
      if (this.isEOF()) {
        return this.error(ErrorKind.EXPECT_ARGUMENT_CLOSING_BRACE, createLocation(openingBracePosition, this.clonePosition()));
      }
      switch (this.char()) {
        // Simple argument: `{name}`
        case 125: {
          this.bump();
          return {
            val: {
              type: TYPE.argument,
              // value does not include the opening and closing braces.
              value,
              location: createLocation(openingBracePosition, this.clonePosition())
            },
            err: null
          };
        }
        // Argument with options: `{name, format, ...}`
        case 44: {
          this.bump();
          this.bumpSpace();
          if (this.isEOF()) {
            return this.error(ErrorKind.EXPECT_ARGUMENT_CLOSING_BRACE, createLocation(openingBracePosition, this.clonePosition()));
          }
          return this.parseArgumentOptions(nestingLevel, expectingCloseTag, value, openingBracePosition);
        }
        default:
          return this.error(ErrorKind.MALFORMED_ARGUMENT, createLocation(openingBracePosition, this.clonePosition()));
      }
    };
    Parser2.prototype.parseIdentifierIfPossible = function() {
      var startingPosition = this.clonePosition();
      var startOffset = this.offset();
      var value = matchIdentifierAtIndex(this.message, startOffset);
      var endOffset = startOffset + value.length;
      this.bumpTo(endOffset);
      var endPosition = this.clonePosition();
      var location2 = createLocation(startingPosition, endPosition);
      return { value, location: location2 };
    };
    Parser2.prototype.parseArgumentOptions = function(nestingLevel, expectingCloseTag, value, openingBracePosition) {
      var _a2;
      var typeStartPosition = this.clonePosition();
      var argType = this.parseIdentifierIfPossible().value;
      var typeEndPosition = this.clonePosition();
      switch (argType) {
        case "":
          return this.error(ErrorKind.EXPECT_ARGUMENT_TYPE, createLocation(typeStartPosition, typeEndPosition));
        case "number":
        case "date":
        case "time": {
          this.bumpSpace();
          var styleAndLocation = null;
          if (this.bumpIf(",")) {
            this.bumpSpace();
            var styleStartPosition = this.clonePosition();
            var result = this.parseSimpleArgStyleIfPossible();
            if (result.err) {
              return result;
            }
            var style = trimEnd(result.val);
            if (style.length === 0) {
              return this.error(ErrorKind.EXPECT_ARGUMENT_STYLE, createLocation(this.clonePosition(), this.clonePosition()));
            }
            var styleLocation = createLocation(styleStartPosition, this.clonePosition());
            styleAndLocation = { style, styleLocation };
          }
          var argCloseResult = this.tryParseArgumentClose(openingBracePosition);
          if (argCloseResult.err) {
            return argCloseResult;
          }
          var location_1 = createLocation(openingBracePosition, this.clonePosition());
          if (styleAndLocation && startsWith(styleAndLocation === null || styleAndLocation === void 0 ? void 0 : styleAndLocation.style, "::", 0)) {
            var skeleton = trimStart(styleAndLocation.style.slice(2));
            if (argType === "number") {
              var result = this.parseNumberSkeletonFromString(skeleton, styleAndLocation.styleLocation);
              if (result.err) {
                return result;
              }
              return {
                val: { type: TYPE.number, value, location: location_1, style: result.val },
                err: null
              };
            } else {
              if (skeleton.length === 0) {
                return this.error(ErrorKind.EXPECT_DATE_TIME_SKELETON, location_1);
              }
              var dateTimePattern = skeleton;
              if (this.locale) {
                dateTimePattern = getBestPattern(skeleton, this.locale);
              }
              var style = {
                type: SKELETON_TYPE.dateTime,
                pattern: dateTimePattern,
                location: styleAndLocation.styleLocation,
                parsedOptions: this.shouldParseSkeletons ? parseDateTimeSkeleton(dateTimePattern) : {}
              };
              var type = argType === "date" ? TYPE.date : TYPE.time;
              return {
                val: { type, value, location: location_1, style },
                err: null
              };
            }
          }
          return {
            val: {
              type: argType === "number" ? TYPE.number : argType === "date" ? TYPE.date : TYPE.time,
              value,
              location: location_1,
              style: (_a2 = styleAndLocation === null || styleAndLocation === void 0 ? void 0 : styleAndLocation.style) !== null && _a2 !== void 0 ? _a2 : null
            },
            err: null
          };
        }
        case "plural":
        case "selectordinal":
        case "select": {
          var typeEndPosition_1 = this.clonePosition();
          this.bumpSpace();
          if (!this.bumpIf(",")) {
            return this.error(ErrorKind.EXPECT_SELECT_ARGUMENT_OPTIONS, createLocation(typeEndPosition_1, __assign({}, typeEndPosition_1)));
          }
          this.bumpSpace();
          var identifierAndLocation = this.parseIdentifierIfPossible();
          var pluralOffset = 0;
          if (argType !== "select" && identifierAndLocation.value === "offset") {
            if (!this.bumpIf(":")) {
              return this.error(ErrorKind.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, createLocation(this.clonePosition(), this.clonePosition()));
            }
            this.bumpSpace();
            var result = this.tryParseDecimalInteger(ErrorKind.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, ErrorKind.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE);
            if (result.err) {
              return result;
            }
            this.bumpSpace();
            identifierAndLocation = this.parseIdentifierIfPossible();
            pluralOffset = result.val;
          }
          var optionsResult = this.tryParsePluralOrSelectOptions(nestingLevel, argType, expectingCloseTag, identifierAndLocation);
          if (optionsResult.err) {
            return optionsResult;
          }
          var argCloseResult = this.tryParseArgumentClose(openingBracePosition);
          if (argCloseResult.err) {
            return argCloseResult;
          }
          var location_2 = createLocation(openingBracePosition, this.clonePosition());
          if (argType === "select") {
            return {
              val: {
                type: TYPE.select,
                value,
                options: fromEntries(optionsResult.val),
                location: location_2
              },
              err: null
            };
          } else {
            return {
              val: {
                type: TYPE.plural,
                value,
                options: fromEntries(optionsResult.val),
                offset: pluralOffset,
                pluralType: argType === "plural" ? "cardinal" : "ordinal",
                location: location_2
              },
              err: null
            };
          }
        }
        default:
          return this.error(ErrorKind.INVALID_ARGUMENT_TYPE, createLocation(typeStartPosition, typeEndPosition));
      }
    };
    Parser2.prototype.tryParseArgumentClose = function(openingBracePosition) {
      if (this.isEOF() || this.char() !== 125) {
        return this.error(ErrorKind.EXPECT_ARGUMENT_CLOSING_BRACE, createLocation(openingBracePosition, this.clonePosition()));
      }
      this.bump();
      return { val: true, err: null };
    };
    Parser2.prototype.parseSimpleArgStyleIfPossible = function() {
      var nestedBraces = 0;
      var startPosition = this.clonePosition();
      while (!this.isEOF()) {
        var ch = this.char();
        switch (ch) {
          case 39: {
            this.bump();
            var apostrophePosition = this.clonePosition();
            if (!this.bumpUntil("'")) {
              return this.error(ErrorKind.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE, createLocation(apostrophePosition, this.clonePosition()));
            }
            this.bump();
            break;
          }
          case 123: {
            nestedBraces += 1;
            this.bump();
            break;
          }
          case 125: {
            if (nestedBraces > 0) {
              nestedBraces -= 1;
            } else {
              return {
                val: this.message.slice(startPosition.offset, this.offset()),
                err: null
              };
            }
            break;
          }
          default:
            this.bump();
            break;
        }
      }
      return {
        val: this.message.slice(startPosition.offset, this.offset()),
        err: null
      };
    };
    Parser2.prototype.parseNumberSkeletonFromString = function(skeleton, location2) {
      var tokens = [];
      try {
        tokens = parseNumberSkeletonFromString(skeleton);
      } catch (e) {
        return this.error(ErrorKind.INVALID_NUMBER_SKELETON, location2);
      }
      return {
        val: {
          type: SKELETON_TYPE.number,
          tokens,
          location: location2,
          parsedOptions: this.shouldParseSkeletons ? parseNumberSkeleton(tokens) : {}
        },
        err: null
      };
    };
    Parser2.prototype.tryParsePluralOrSelectOptions = function(nestingLevel, parentArgType, expectCloseTag, parsedFirstIdentifier) {
      var _a2;
      var hasOtherClause = false;
      var options2 = [];
      var parsedSelectors = /* @__PURE__ */ new Set();
      var selector = parsedFirstIdentifier.value, selectorLocation = parsedFirstIdentifier.location;
      while (true) {
        if (selector.length === 0) {
          var startPosition = this.clonePosition();
          if (parentArgType !== "select" && this.bumpIf("=")) {
            var result = this.tryParseDecimalInteger(ErrorKind.EXPECT_PLURAL_ARGUMENT_SELECTOR, ErrorKind.INVALID_PLURAL_ARGUMENT_SELECTOR);
            if (result.err) {
              return result;
            }
            selectorLocation = createLocation(startPosition, this.clonePosition());
            selector = this.message.slice(startPosition.offset, this.offset());
          } else {
            break;
          }
        }
        if (parsedSelectors.has(selector)) {
          return this.error(parentArgType === "select" ? ErrorKind.DUPLICATE_SELECT_ARGUMENT_SELECTOR : ErrorKind.DUPLICATE_PLURAL_ARGUMENT_SELECTOR, selectorLocation);
        }
        if (selector === "other") {
          hasOtherClause = true;
        }
        this.bumpSpace();
        var openingBracePosition = this.clonePosition();
        if (!this.bumpIf("{")) {
          return this.error(parentArgType === "select" ? ErrorKind.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT : ErrorKind.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT, createLocation(this.clonePosition(), this.clonePosition()));
        }
        var fragmentResult = this.parseMessage(nestingLevel + 1, parentArgType, expectCloseTag);
        if (fragmentResult.err) {
          return fragmentResult;
        }
        var argCloseResult = this.tryParseArgumentClose(openingBracePosition);
        if (argCloseResult.err) {
          return argCloseResult;
        }
        options2.push([
          selector,
          {
            value: fragmentResult.val,
            location: createLocation(openingBracePosition, this.clonePosition())
          }
        ]);
        parsedSelectors.add(selector);
        this.bumpSpace();
        _a2 = this.parseIdentifierIfPossible(), selector = _a2.value, selectorLocation = _a2.location;
      }
      if (options2.length === 0) {
        return this.error(parentArgType === "select" ? ErrorKind.EXPECT_SELECT_ARGUMENT_SELECTOR : ErrorKind.EXPECT_PLURAL_ARGUMENT_SELECTOR, createLocation(this.clonePosition(), this.clonePosition()));
      }
      if (this.requiresOtherClause && !hasOtherClause) {
        return this.error(ErrorKind.MISSING_OTHER_CLAUSE, createLocation(this.clonePosition(), this.clonePosition()));
      }
      return { val: options2, err: null };
    };
    Parser2.prototype.tryParseDecimalInteger = function(expectNumberError, invalidNumberError) {
      var sign = 1;
      var startingPosition = this.clonePosition();
      if (this.bumpIf("+")) ;
      else if (this.bumpIf("-")) {
        sign = -1;
      }
      var hasDigits = false;
      var decimal = 0;
      while (!this.isEOF()) {
        var ch = this.char();
        if (ch >= 48 && ch <= 57) {
          hasDigits = true;
          decimal = decimal * 10 + (ch - 48);
          this.bump();
        } else {
          break;
        }
      }
      var location2 = createLocation(startingPosition, this.clonePosition());
      if (!hasDigits) {
        return this.error(expectNumberError, location2);
      }
      decimal *= sign;
      if (!isSafeInteger(decimal)) {
        return this.error(invalidNumberError, location2);
      }
      return { val: decimal, err: null };
    };
    Parser2.prototype.offset = function() {
      return this.position.offset;
    };
    Parser2.prototype.isEOF = function() {
      return this.offset() === this.message.length;
    };
    Parser2.prototype.clonePosition = function() {
      return {
        offset: this.position.offset,
        line: this.position.line,
        column: this.position.column
      };
    };
    Parser2.prototype.char = function() {
      var offset = this.position.offset;
      if (offset >= this.message.length) {
        throw Error("out of bound");
      }
      var code2 = codePointAt(this.message, offset);
      if (code2 === void 0) {
        throw Error("Offset ".concat(offset, " is at invalid UTF-16 code unit boundary"));
      }
      return code2;
    };
    Parser2.prototype.error = function(kind, location2) {
      return {
        val: null,
        err: {
          kind,
          message: this.message,
          location: location2
        }
      };
    };
    Parser2.prototype.bump = function() {
      if (this.isEOF()) {
        return;
      }
      var code2 = this.char();
      if (code2 === 10) {
        this.position.line += 1;
        this.position.column = 1;
        this.position.offset += 1;
      } else {
        this.position.column += 1;
        this.position.offset += code2 < 65536 ? 1 : 2;
      }
    };
    Parser2.prototype.bumpIf = function(prefix) {
      if (startsWith(this.message, prefix, this.offset())) {
        for (var i = 0; i < prefix.length; i++) {
          this.bump();
        }
        return true;
      }
      return false;
    };
    Parser2.prototype.bumpUntil = function(pattern) {
      var currentOffset = this.offset();
      var index = this.message.indexOf(pattern, currentOffset);
      if (index >= 0) {
        this.bumpTo(index);
        return true;
      } else {
        this.bumpTo(this.message.length);
        return false;
      }
    };
    Parser2.prototype.bumpTo = function(targetOffset) {
      if (this.offset() > targetOffset) {
        throw Error("targetOffset ".concat(targetOffset, " must be greater than or equal to the current offset ").concat(this.offset()));
      }
      targetOffset = Math.min(targetOffset, this.message.length);
      while (true) {
        var offset = this.offset();
        if (offset === targetOffset) {
          break;
        }
        if (offset > targetOffset) {
          throw Error("targetOffset ".concat(targetOffset, " is at invalid UTF-16 code unit boundary"));
        }
        this.bump();
        if (this.isEOF()) {
          break;
        }
      }
    };
    Parser2.prototype.bumpSpace = function() {
      while (!this.isEOF() && _isWhiteSpace(this.char())) {
        this.bump();
      }
    };
    Parser2.prototype.peek = function() {
      if (this.isEOF()) {
        return null;
      }
      var code2 = this.char();
      var offset = this.offset();
      var nextCode = this.message.charCodeAt(offset + (code2 >= 65536 ? 2 : 1));
      return nextCode !== null && nextCode !== void 0 ? nextCode : null;
    };
    return Parser2;
  })()
);
function _isAlpha(codepoint) {
  return codepoint >= 97 && codepoint <= 122 || codepoint >= 65 && codepoint <= 90;
}
function _isAlphaOrSlash(codepoint) {
  return _isAlpha(codepoint) || codepoint === 47;
}
function _isPotentialElementNameChar(c) {
  return c === 45 || c === 46 || c >= 48 && c <= 57 || c === 95 || c >= 97 && c <= 122 || c >= 65 && c <= 90 || c == 183 || c >= 192 && c <= 214 || c >= 216 && c <= 246 || c >= 248 && c <= 893 || c >= 895 && c <= 8191 || c >= 8204 && c <= 8205 || c >= 8255 && c <= 8256 || c >= 8304 && c <= 8591 || c >= 11264 && c <= 12271 || c >= 12289 && c <= 55295 || c >= 63744 && c <= 64975 || c >= 65008 && c <= 65533 || c >= 65536 && c <= 983039;
}
function _isWhiteSpace(c) {
  return c >= 9 && c <= 13 || c === 32 || c === 133 || c >= 8206 && c <= 8207 || c === 8232 || c === 8233;
}
function _isPatternSyntax(c) {
  return c >= 33 && c <= 35 || c === 36 || c >= 37 && c <= 39 || c === 40 || c === 41 || c === 42 || c === 43 || c === 44 || c === 45 || c >= 46 && c <= 47 || c >= 58 && c <= 59 || c >= 60 && c <= 62 || c >= 63 && c <= 64 || c === 91 || c === 92 || c === 93 || c === 94 || c === 96 || c === 123 || c === 124 || c === 125 || c === 126 || c === 161 || c >= 162 && c <= 165 || c === 166 || c === 167 || c === 169 || c === 171 || c === 172 || c === 174 || c === 176 || c === 177 || c === 182 || c === 187 || c === 191 || c === 215 || c === 247 || c >= 8208 && c <= 8213 || c >= 8214 && c <= 8215 || c === 8216 || c === 8217 || c === 8218 || c >= 8219 && c <= 8220 || c === 8221 || c === 8222 || c === 8223 || c >= 8224 && c <= 8231 || c >= 8240 && c <= 8248 || c === 8249 || c === 8250 || c >= 8251 && c <= 8254 || c >= 8257 && c <= 8259 || c === 8260 || c === 8261 || c === 8262 || c >= 8263 && c <= 8273 || c === 8274 || c === 8275 || c >= 8277 && c <= 8286 || c >= 8592 && c <= 8596 || c >= 8597 && c <= 8601 || c >= 8602 && c <= 8603 || c >= 8604 && c <= 8607 || c === 8608 || c >= 8609 && c <= 8610 || c === 8611 || c >= 8612 && c <= 8613 || c === 8614 || c >= 8615 && c <= 8621 || c === 8622 || c >= 8623 && c <= 8653 || c >= 8654 && c <= 8655 || c >= 8656 && c <= 8657 || c === 8658 || c === 8659 || c === 8660 || c >= 8661 && c <= 8691 || c >= 8692 && c <= 8959 || c >= 8960 && c <= 8967 || c === 8968 || c === 8969 || c === 8970 || c === 8971 || c >= 8972 && c <= 8991 || c >= 8992 && c <= 8993 || c >= 8994 && c <= 9e3 || c === 9001 || c === 9002 || c >= 9003 && c <= 9083 || c === 9084 || c >= 9085 && c <= 9114 || c >= 9115 && c <= 9139 || c >= 9140 && c <= 9179 || c >= 9180 && c <= 9185 || c >= 9186 && c <= 9254 || c >= 9255 && c <= 9279 || c >= 9280 && c <= 9290 || c >= 9291 && c <= 9311 || c >= 9472 && c <= 9654 || c === 9655 || c >= 9656 && c <= 9664 || c === 9665 || c >= 9666 && c <= 9719 || c >= 9720 && c <= 9727 || c >= 9728 && c <= 9838 || c === 9839 || c >= 9840 && c <= 10087 || c === 10088 || c === 10089 || c === 10090 || c === 10091 || c === 10092 || c === 10093 || c === 10094 || c === 10095 || c === 10096 || c === 10097 || c === 10098 || c === 10099 || c === 10100 || c === 10101 || c >= 10132 && c <= 10175 || c >= 10176 && c <= 10180 || c === 10181 || c === 10182 || c >= 10183 && c <= 10213 || c === 10214 || c === 10215 || c === 10216 || c === 10217 || c === 10218 || c === 10219 || c === 10220 || c === 10221 || c === 10222 || c === 10223 || c >= 10224 && c <= 10239 || c >= 10240 && c <= 10495 || c >= 10496 && c <= 10626 || c === 10627 || c === 10628 || c === 10629 || c === 10630 || c === 10631 || c === 10632 || c === 10633 || c === 10634 || c === 10635 || c === 10636 || c === 10637 || c === 10638 || c === 10639 || c === 10640 || c === 10641 || c === 10642 || c === 10643 || c === 10644 || c === 10645 || c === 10646 || c === 10647 || c === 10648 || c >= 10649 && c <= 10711 || c === 10712 || c === 10713 || c === 10714 || c === 10715 || c >= 10716 && c <= 10747 || c === 10748 || c === 10749 || c >= 10750 && c <= 11007 || c >= 11008 && c <= 11055 || c >= 11056 && c <= 11076 || c >= 11077 && c <= 11078 || c >= 11079 && c <= 11084 || c >= 11085 && c <= 11123 || c >= 11124 && c <= 11125 || c >= 11126 && c <= 11157 || c === 11158 || c >= 11159 && c <= 11263 || c >= 11776 && c <= 11777 || c === 11778 || c === 11779 || c === 11780 || c === 11781 || c >= 11782 && c <= 11784 || c === 11785 || c === 11786 || c === 11787 || c === 11788 || c === 11789 || c >= 11790 && c <= 11798 || c === 11799 || c >= 11800 && c <= 11801 || c === 11802 || c === 11803 || c === 11804 || c === 11805 || c >= 11806 && c <= 11807 || c === 11808 || c === 11809 || c === 11810 || c === 11811 || c === 11812 || c === 11813 || c === 11814 || c === 11815 || c === 11816 || c === 11817 || c >= 11818 && c <= 11822 || c === 11823 || c >= 11824 && c <= 11833 || c >= 11834 && c <= 11835 || c >= 11836 && c <= 11839 || c === 11840 || c === 11841 || c === 11842 || c >= 11843 && c <= 11855 || c >= 11856 && c <= 11857 || c === 11858 || c >= 11859 && c <= 11903 || c >= 12289 && c <= 12291 || c === 12296 || c === 12297 || c === 12298 || c === 12299 || c === 12300 || c === 12301 || c === 12302 || c === 12303 || c === 12304 || c === 12305 || c >= 12306 && c <= 12307 || c === 12308 || c === 12309 || c === 12310 || c === 12311 || c === 12312 || c === 12313 || c === 12314 || c === 12315 || c === 12316 || c === 12317 || c >= 12318 && c <= 12319 || c === 12320 || c === 12336 || c === 64830 || c === 64831 || c >= 65093 && c <= 65094;
}
function pruneLocation(els) {
  els.forEach(function(el) {
    delete el.location;
    if (isSelectElement(el) || isPluralElement(el)) {
      for (var k in el.options) {
        delete el.options[k].location;
        pruneLocation(el.options[k].value);
      }
    } else if (isNumberElement(el) && isNumberSkeleton(el.style)) {
      delete el.style.location;
    } else if ((isDateElement(el) || isTimeElement(el)) && isDateTimeSkeleton(el.style)) {
      delete el.style.location;
    } else if (isTagElement(el)) {
      pruneLocation(el.children);
    }
  });
}
function parse(message, opts) {
  if (opts === void 0) {
    opts = {};
  }
  opts = __assign({ shouldParseSkeletons: true, requiresOtherClause: true }, opts);
  var result = new Parser(message, opts).parse();
  if (result.err) {
    var error = SyntaxError(ErrorKind[result.err.kind]);
    error.location = result.err.location;
    error.originalMessage = result.err.message;
    throw error;
  }
  if (!(opts === null || opts === void 0 ? void 0 : opts.captureLocation)) {
    pruneLocation(result.val);
  }
  return result.val;
}
var ErrorCode;
(function(ErrorCode2) {
  ErrorCode2["MISSING_VALUE"] = "MISSING_VALUE";
  ErrorCode2["INVALID_VALUE"] = "INVALID_VALUE";
  ErrorCode2["MISSING_INTL_API"] = "MISSING_INTL_API";
})(ErrorCode || (ErrorCode = {}));
var FormatError = (
  /** @class */
  (function(_super) {
    __extends(FormatError2, _super);
    function FormatError2(msg, code2, originalMessage) {
      var _this = _super.call(this, msg) || this;
      _this.code = code2;
      _this.originalMessage = originalMessage;
      return _this;
    }
    FormatError2.prototype.toString = function() {
      return "[formatjs Error: ".concat(this.code, "] ").concat(this.message);
    };
    return FormatError2;
  })(Error)
);
var InvalidValueError = (
  /** @class */
  (function(_super) {
    __extends(InvalidValueError2, _super);
    function InvalidValueError2(variableId, value, options2, originalMessage) {
      return _super.call(this, 'Invalid values for "'.concat(variableId, '": "').concat(value, '". Options are "').concat(Object.keys(options2).join('", "'), '"'), ErrorCode.INVALID_VALUE, originalMessage) || this;
    }
    return InvalidValueError2;
  })(FormatError)
);
var InvalidValueTypeError = (
  /** @class */
  (function(_super) {
    __extends(InvalidValueTypeError2, _super);
    function InvalidValueTypeError2(value, type, originalMessage) {
      return _super.call(this, 'Value for "'.concat(value, '" must be of type ').concat(type), ErrorCode.INVALID_VALUE, originalMessage) || this;
    }
    return InvalidValueTypeError2;
  })(FormatError)
);
var MissingValueError = (
  /** @class */
  (function(_super) {
    __extends(MissingValueError2, _super);
    function MissingValueError2(variableId, originalMessage) {
      return _super.call(this, 'The intl string context variable "'.concat(variableId, '" was not provided to the string "').concat(originalMessage, '"'), ErrorCode.MISSING_VALUE, originalMessage) || this;
    }
    return MissingValueError2;
  })(FormatError)
);
var PART_TYPE;
(function(PART_TYPE2) {
  PART_TYPE2[PART_TYPE2["literal"] = 0] = "literal";
  PART_TYPE2[PART_TYPE2["object"] = 1] = "object";
})(PART_TYPE || (PART_TYPE = {}));
function mergeLiteral(parts) {
  if (parts.length < 2) {
    return parts;
  }
  return parts.reduce(function(all, part) {
    var lastPart = all[all.length - 1];
    if (!lastPart || lastPart.type !== PART_TYPE.literal || part.type !== PART_TYPE.literal) {
      all.push(part);
    } else {
      lastPart.value += part.value;
    }
    return all;
  }, []);
}
function isFormatXMLElementFn(el) {
  return typeof el === "function";
}
function formatToParts(els, locales, formatters, formats, values, currentPluralValue, originalMessage) {
  if (els.length === 1 && isLiteralElement(els[0])) {
    return [
      {
        type: PART_TYPE.literal,
        value: els[0].value
      }
    ];
  }
  var result = [];
  for (var _i = 0, els_1 = els; _i < els_1.length; _i++) {
    var el = els_1[_i];
    if (isLiteralElement(el)) {
      result.push({
        type: PART_TYPE.literal,
        value: el.value
      });
      continue;
    }
    if (isPoundElement(el)) {
      if (typeof currentPluralValue === "number") {
        result.push({
          type: PART_TYPE.literal,
          value: formatters.getNumberFormat(locales).format(currentPluralValue)
        });
      }
      continue;
    }
    var varName = el.value;
    if (!(values && varName in values)) {
      throw new MissingValueError(varName, originalMessage);
    }
    var value = values[varName];
    if (isArgumentElement(el)) {
      if (!value || typeof value === "string" || typeof value === "number") {
        value = typeof value === "string" || typeof value === "number" ? String(value) : "";
      }
      result.push({
        type: typeof value === "string" ? PART_TYPE.literal : PART_TYPE.object,
        value
      });
      continue;
    }
    if (isDateElement(el)) {
      var style = typeof el.style === "string" ? formats.date[el.style] : isDateTimeSkeleton(el.style) ? el.style.parsedOptions : void 0;
      result.push({
        type: PART_TYPE.literal,
        value: formatters.getDateTimeFormat(locales, style).format(value)
      });
      continue;
    }
    if (isTimeElement(el)) {
      var style = typeof el.style === "string" ? formats.time[el.style] : isDateTimeSkeleton(el.style) ? el.style.parsedOptions : formats.time.medium;
      result.push({
        type: PART_TYPE.literal,
        value: formatters.getDateTimeFormat(locales, style).format(value)
      });
      continue;
    }
    if (isNumberElement(el)) {
      var style = typeof el.style === "string" ? formats.number[el.style] : isNumberSkeleton(el.style) ? el.style.parsedOptions : void 0;
      if (style && style.scale) {
        value = value * (style.scale || 1);
      }
      result.push({
        type: PART_TYPE.literal,
        value: formatters.getNumberFormat(locales, style).format(value)
      });
      continue;
    }
    if (isTagElement(el)) {
      var children = el.children, value_1 = el.value;
      var formatFn = values[value_1];
      if (!isFormatXMLElementFn(formatFn)) {
        throw new InvalidValueTypeError(value_1, "function", originalMessage);
      }
      var parts = formatToParts(children, locales, formatters, formats, values, currentPluralValue);
      var chunks = formatFn(parts.map(function(p) {
        return p.value;
      }));
      if (!Array.isArray(chunks)) {
        chunks = [chunks];
      }
      result.push.apply(result, chunks.map(function(c) {
        return {
          type: typeof c === "string" ? PART_TYPE.literal : PART_TYPE.object,
          value: c
        };
      }));
    }
    if (isSelectElement(el)) {
      var opt = el.options[value] || el.options.other;
      if (!opt) {
        throw new InvalidValueError(el.value, value, Object.keys(el.options), originalMessage);
      }
      result.push.apply(result, formatToParts(opt.value, locales, formatters, formats, values));
      continue;
    }
    if (isPluralElement(el)) {
      var opt = el.options["=".concat(value)];
      if (!opt) {
        if (!Intl.PluralRules) {
          throw new FormatError('Intl.PluralRules is not available in this environment.\nTry polyfilling it using "@formatjs/intl-pluralrules"\n', ErrorCode.MISSING_INTL_API, originalMessage);
        }
        var rule = formatters.getPluralRules(locales, { type: el.pluralType }).select(value - (el.offset || 0));
        opt = el.options[rule] || el.options.other;
      }
      if (!opt) {
        throw new InvalidValueError(el.value, value, Object.keys(el.options), originalMessage);
      }
      result.push.apply(result, formatToParts(opt.value, locales, formatters, formats, values, value - (el.offset || 0)));
      continue;
    }
  }
  return mergeLiteral(result);
}
function mergeConfig(c1, c2) {
  if (!c2) {
    return c1;
  }
  return __assign(__assign(__assign({}, c1 || {}), c2 || {}), Object.keys(c1).reduce(function(all, k) {
    all[k] = __assign(__assign({}, c1[k]), c2[k] || {});
    return all;
  }, {}));
}
function mergeConfigs(defaultConfig, configs) {
  if (!configs) {
    return defaultConfig;
  }
  return Object.keys(defaultConfig).reduce(function(all, k) {
    all[k] = mergeConfig(defaultConfig[k], configs[k]);
    return all;
  }, __assign({}, defaultConfig));
}
function createFastMemoizeCache(store) {
  return {
    create: function() {
      return {
        get: function(key) {
          return store[key];
        },
        set: function(key, value) {
          store[key] = value;
        }
      };
    }
  };
}
function createDefaultFormatters(cache) {
  if (cache === void 0) {
    cache = {
      number: {},
      dateTime: {},
      pluralRules: {}
    };
  }
  return {
    getNumberFormat: memoize(function() {
      var _a2;
      var args = [];
      for (var _i = 0; _i < arguments.length; _i++) {
        args[_i] = arguments[_i];
      }
      return new ((_a2 = Intl.NumberFormat).bind.apply(_a2, __spreadArray([void 0], args, false)))();
    }, {
      cache: createFastMemoizeCache(cache.number),
      strategy: strategies.variadic
    }),
    getDateTimeFormat: memoize(function() {
      var _a2;
      var args = [];
      for (var _i = 0; _i < arguments.length; _i++) {
        args[_i] = arguments[_i];
      }
      return new ((_a2 = Intl.DateTimeFormat).bind.apply(_a2, __spreadArray([void 0], args, false)))();
    }, {
      cache: createFastMemoizeCache(cache.dateTime),
      strategy: strategies.variadic
    }),
    getPluralRules: memoize(function() {
      var _a2;
      var args = [];
      for (var _i = 0; _i < arguments.length; _i++) {
        args[_i] = arguments[_i];
      }
      return new ((_a2 = Intl.PluralRules).bind.apply(_a2, __spreadArray([void 0], args, false)))();
    }, {
      cache: createFastMemoizeCache(cache.pluralRules),
      strategy: strategies.variadic
    })
  };
}
var IntlMessageFormat = (
  /** @class */
  (function() {
    function IntlMessageFormat2(message, locales, overrideFormats, opts) {
      if (locales === void 0) {
        locales = IntlMessageFormat2.defaultLocale;
      }
      var _this = this;
      this.formatterCache = {
        number: {},
        dateTime: {},
        pluralRules: {}
      };
      this.format = function(values) {
        var parts = _this.formatToParts(values);
        if (parts.length === 1) {
          return parts[0].value;
        }
        var result = parts.reduce(function(all, part) {
          if (!all.length || part.type !== PART_TYPE.literal || typeof all[all.length - 1] !== "string") {
            all.push(part.value);
          } else {
            all[all.length - 1] += part.value;
          }
          return all;
        }, []);
        if (result.length <= 1) {
          return result[0] || "";
        }
        return result;
      };
      this.formatToParts = function(values) {
        return formatToParts(_this.ast, _this.locales, _this.formatters, _this.formats, values, void 0, _this.message);
      };
      this.resolvedOptions = function() {
        var _a3;
        return {
          locale: ((_a3 = _this.resolvedLocale) === null || _a3 === void 0 ? void 0 : _a3.toString()) || Intl.NumberFormat.supportedLocalesOf(_this.locales)[0]
        };
      };
      this.getAst = function() {
        return _this.ast;
      };
      this.locales = locales;
      this.resolvedLocale = IntlMessageFormat2.resolveLocale(locales);
      if (typeof message === "string") {
        this.message = message;
        if (!IntlMessageFormat2.__parse) {
          throw new TypeError("IntlMessageFormat.__parse must be set to process `message` of type `string`");
        }
        var _a2 = opts || {};
        _a2.formatters;
        var parseOpts = __rest(_a2, ["formatters"]);
        this.ast = IntlMessageFormat2.__parse(message, __assign(__assign({}, parseOpts), { locale: this.resolvedLocale }));
      } else {
        this.ast = message;
      }
      if (!Array.isArray(this.ast)) {
        throw new TypeError("A message must be provided as a String or AST.");
      }
      this.formats = mergeConfigs(IntlMessageFormat2.formats, overrideFormats);
      this.formatters = opts && opts.formatters || createDefaultFormatters(this.formatterCache);
    }
    Object.defineProperty(IntlMessageFormat2, "defaultLocale", {
      get: function() {
        if (!IntlMessageFormat2.memoizedDefaultLocale) {
          IntlMessageFormat2.memoizedDefaultLocale = new Intl.NumberFormat().resolvedOptions().locale;
        }
        return IntlMessageFormat2.memoizedDefaultLocale;
      },
      enumerable: false,
      configurable: true
    });
    IntlMessageFormat2.memoizedDefaultLocale = null;
    IntlMessageFormat2.resolveLocale = function(locales) {
      if (typeof Intl.Locale === "undefined") {
        return;
      }
      var supportedLocales = Intl.NumberFormat.supportedLocalesOf(locales);
      if (supportedLocales.length > 0) {
        return new Intl.Locale(supportedLocales[0]);
      }
      return new Intl.Locale(typeof locales === "string" ? locales : locales[0]);
    };
    IntlMessageFormat2.__parse = parse;
    IntlMessageFormat2.formats = {
      number: {
        integer: {
          maximumFractionDigits: 0
        },
        currency: {
          style: "currency"
        },
        percent: {
          style: "percent"
        }
      },
      date: {
        short: {
          month: "numeric",
          day: "numeric",
          year: "2-digit"
        },
        medium: {
          month: "short",
          day: "numeric",
          year: "numeric"
        },
        long: {
          month: "long",
          day: "numeric",
          year: "numeric"
        },
        full: {
          weekday: "long",
          month: "long",
          day: "numeric",
          year: "numeric"
        }
      },
      time: {
        short: {
          hour: "numeric",
          minute: "numeric"
        },
        medium: {
          hour: "numeric",
          minute: "numeric",
          second: "numeric"
        },
        long: {
          hour: "numeric",
          minute: "numeric",
          second: "numeric",
          timeZoneName: "short"
        },
        full: {
          hour: "numeric",
          minute: "numeric",
          second: "numeric",
          timeZoneName: "short"
        }
      }
    };
    return IntlMessageFormat2;
  })()
);
function delve(obj, fullKey) {
  if (fullKey == null)
    return void 0;
  if (fullKey in obj) {
    return obj[fullKey];
  }
  const keys = fullKey.split(".");
  let result = obj;
  for (let p = 0; p < keys.length; p++) {
    if (typeof result === "object") {
      if (p > 0) {
        const partialKey = keys.slice(p, keys.length).join(".");
        if (partialKey in result) {
          result = result[partialKey];
          break;
        }
      }
      result = result[keys[p]];
    } else {
      result = void 0;
    }
  }
  return result;
}
const lookupCache = {};
const addToCache = (path, locale, message) => {
  if (!message)
    return message;
  if (!(locale in lookupCache))
    lookupCache[locale] = {};
  if (!(path in lookupCache[locale]))
    lookupCache[locale][path] = message;
  return message;
};
const lookup = (path, refLocale) => {
  if (refLocale == null)
    return void 0;
  if (refLocale in lookupCache && path in lookupCache[refLocale]) {
    return lookupCache[refLocale][path];
  }
  const locales = getPossibleLocales(refLocale);
  for (let i = 0; i < locales.length; i++) {
    const locale = locales[i];
    const message = getMessageFromDictionary(locale, path);
    if (message) {
      return addToCache(path, refLocale, message);
    }
  }
  return void 0;
};
let dictionary;
const $dictionary = writable({});
function getLocaleDictionary(locale) {
  return dictionary[locale] || null;
}
function hasLocaleDictionary(locale) {
  return locale in dictionary;
}
function getMessageFromDictionary(locale, id) {
  if (!hasLocaleDictionary(locale)) {
    return null;
  }
  const localeDictionary = getLocaleDictionary(locale);
  const match = delve(localeDictionary, id);
  return match;
}
function getClosestAvailableLocale(refLocale) {
  if (refLocale == null)
    return void 0;
  const relatedLocales = getPossibleLocales(refLocale);
  for (let i = 0; i < relatedLocales.length; i++) {
    const locale = relatedLocales[i];
    if (hasLocaleDictionary(locale)) {
      return locale;
    }
  }
  return void 0;
}
function addMessages(locale, ...partials) {
  delete lookupCache[locale];
  $dictionary.update((d) => {
    d[locale] = deepmerge.all([d[locale] || {}, ...partials]);
    return d;
  });
}
derived(
  [$dictionary],
  ([dictionary2]) => Object.keys(dictionary2)
);
$dictionary.subscribe((newDictionary) => dictionary = newDictionary);
const queue = {};
function createLocaleQueue(locale) {
  queue[locale] = /* @__PURE__ */ new Set();
}
function removeLoaderFromQueue(locale, loader) {
  queue[locale].delete(loader);
  if (queue[locale].size === 0) {
    delete queue[locale];
  }
}
function getLocaleQueue(locale) {
  return queue[locale];
}
function getLocalesQueues(locale) {
  return getPossibleLocales(locale).map((localeItem) => {
    const localeQueue = getLocaleQueue(localeItem);
    return [localeItem, localeQueue ? [...localeQueue] : []];
  }).filter(([, localeQueue]) => localeQueue.length > 0);
}
function hasLocaleQueue(locale) {
  if (locale == null)
    return false;
  return getPossibleLocales(locale).some(
    (localeQueue) => {
      var _a2;
      return (_a2 = getLocaleQueue(localeQueue)) == null ? void 0 : _a2.size;
    }
  );
}
function loadLocaleQueue(locale, localeQueue) {
  const allLoadersPromise = Promise.all(
    localeQueue.map((loader) => {
      removeLoaderFromQueue(locale, loader);
      return loader().then((partial) => partial.default || partial);
    })
  );
  return allLoadersPromise.then((partials) => addMessages(locale, ...partials));
}
const activeFlushes = {};
function flush(locale) {
  if (!hasLocaleQueue(locale)) {
    if (locale in activeFlushes) {
      return activeFlushes[locale];
    }
    return Promise.resolve();
  }
  const queues = getLocalesQueues(locale);
  activeFlushes[locale] = Promise.all(
    queues.map(
      ([localeName, localeQueue]) => loadLocaleQueue(localeName, localeQueue)
    )
  ).then(() => {
    if (hasLocaleQueue(locale)) {
      return flush(locale);
    }
    delete activeFlushes[locale];
  });
  return activeFlushes[locale];
}
function registerLocaleLoader(locale, loader) {
  if (!getLocaleQueue(locale))
    createLocaleQueue(locale);
  const localeQueue = getLocaleQueue(locale);
  if (getLocaleQueue(locale).has(loader))
    return;
  if (!hasLocaleDictionary(locale)) {
    $dictionary.update((d) => {
      d[locale] = {};
      return d;
    });
  }
  localeQueue.add(loader);
}
var __getOwnPropSymbols$2 = Object.getOwnPropertySymbols;
var __hasOwnProp$2 = Object.prototype.hasOwnProperty;
var __propIsEnum$2 = Object.prototype.propertyIsEnumerable;
var __objRest$1 = (source, exclude) => {
  var target = {};
  for (var prop in source)
    if (__hasOwnProp$2.call(source, prop) && exclude.indexOf(prop) < 0)
      target[prop] = source[prop];
  if (source != null && __getOwnPropSymbols$2)
    for (var prop of __getOwnPropSymbols$2(source)) {
      if (exclude.indexOf(prop) < 0 && __propIsEnum$2.call(source, prop))
        target[prop] = source[prop];
    }
  return target;
};
const defaultFormats = {
  number: {
    scientific: { notation: "scientific" },
    engineering: { notation: "engineering" },
    compactLong: { notation: "compact", compactDisplay: "long" },
    compactShort: { notation: "compact", compactDisplay: "short" }
  },
  date: {
    short: { month: "numeric", day: "numeric", year: "2-digit" },
    medium: { month: "short", day: "numeric", year: "numeric" },
    long: { month: "long", day: "numeric", year: "numeric" },
    full: { weekday: "long", month: "long", day: "numeric", year: "numeric" }
  },
  time: {
    short: { hour: "numeric", minute: "numeric" },
    medium: { hour: "numeric", minute: "numeric", second: "numeric" },
    long: {
      hour: "numeric",
      minute: "numeric",
      second: "numeric",
      timeZoneName: "short"
    },
    full: {
      hour: "numeric",
      minute: "numeric",
      second: "numeric",
      timeZoneName: "short"
    }
  }
};
function defaultMissingKeyHandler({ locale, id }) {
  console.warn(
    `[svelte-i18n] The message "${id}" was not found in "${getPossibleLocales(
      locale
    ).join('", "')}".${hasLocaleQueue(getCurrentLocale()) ? `

Note: there are at least one loader still registered to this locale that wasn't executed.` : ""}`
  );
}
const defaultOptions = {
  fallbackLocale: null,
  loadingDelay: 200,
  formats: defaultFormats,
  warnOnMissingMessages: true,
  handleMissingMessage: void 0,
  ignoreTag: true
};
const options = defaultOptions;
function getOptions() {
  return options;
}
function init(opts) {
  const _a2 = opts, { formats } = _a2, rest = __objRest$1(_a2, ["formats"]);
  let initialLocale = opts.fallbackLocale;
  if (opts.initialLocale) {
    try {
      if (IntlMessageFormat.resolveLocale(opts.initialLocale)) {
        initialLocale = opts.initialLocale;
      }
    } catch (e) {
      console.warn(
        `[svelte-i18n] The initial locale "${opts.initialLocale}" is not a valid locale.`
      );
    }
  }
  if (rest.warnOnMissingMessages) {
    delete rest.warnOnMissingMessages;
    if (rest.handleMissingMessage == null) {
      rest.handleMissingMessage = defaultMissingKeyHandler;
    } else {
      console.warn(
        '[svelte-i18n] The "warnOnMissingMessages" option is deprecated. Please use the "handleMissingMessage" option instead.'
      );
    }
  }
  Object.assign(options, rest, { initialLocale });
  if (formats) {
    if ("number" in formats) {
      Object.assign(options.formats.number, formats.number);
    }
    if ("date" in formats) {
      Object.assign(options.formats.date, formats.date);
    }
    if ("time" in formats) {
      Object.assign(options.formats.time, formats.time);
    }
  }
  return $locale.set(initialLocale);
}
const $isLoading = writable(false);
var __defProp$1 = Object.defineProperty;
var __defProps = Object.defineProperties;
var __getOwnPropDescs = Object.getOwnPropertyDescriptors;
var __getOwnPropSymbols$1 = Object.getOwnPropertySymbols;
var __hasOwnProp$1 = Object.prototype.hasOwnProperty;
var __propIsEnum$1 = Object.prototype.propertyIsEnumerable;
var __defNormalProp$1 = (obj, key, value) => key in obj ? __defProp$1(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __spreadValues$1 = (a, b) => {
  for (var prop in b || (b = {}))
    if (__hasOwnProp$1.call(b, prop))
      __defNormalProp$1(a, prop, b[prop]);
  if (__getOwnPropSymbols$1)
    for (var prop of __getOwnPropSymbols$1(b)) {
      if (__propIsEnum$1.call(b, prop))
        __defNormalProp$1(a, prop, b[prop]);
    }
  return a;
};
var __spreadProps = (a, b) => __defProps(a, __getOwnPropDescs(b));
let current;
const internalLocale = writable(null);
function getSubLocales(refLocale) {
  return refLocale.split("-").map((_, i, arr) => arr.slice(0, i + 1).join("-")).reverse();
}
function getPossibleLocales(refLocale, fallbackLocale = getOptions().fallbackLocale) {
  const locales = getSubLocales(refLocale);
  if (fallbackLocale) {
    return [.../* @__PURE__ */ new Set([...locales, ...getSubLocales(fallbackLocale)])];
  }
  return locales;
}
function getCurrentLocale() {
  return current != null ? current : void 0;
}
internalLocale.subscribe((newLocale) => {
  current = newLocale != null ? newLocale : void 0;
  if (typeof window !== "undefined" && newLocale != null) {
    document.documentElement.setAttribute("lang", newLocale);
  }
});
const set = (newLocale) => {
  if (newLocale && getClosestAvailableLocale(newLocale) && hasLocaleQueue(newLocale)) {
    const { loadingDelay } = getOptions();
    let loadingTimer;
    if (typeof window !== "undefined" && getCurrentLocale() != null && loadingDelay) {
      loadingTimer = window.setTimeout(
        () => $isLoading.set(true),
        loadingDelay
      );
    } else {
      $isLoading.set(true);
    }
    return flush(newLocale).then(() => {
      internalLocale.set(newLocale);
    }).finally(() => {
      clearTimeout(loadingTimer);
      $isLoading.set(false);
    });
  }
  return internalLocale.set(newLocale);
};
const $locale = __spreadProps(__spreadValues$1({}, internalLocale), {
  set
});
const getLocaleFromNavigator = () => {
  if (typeof window === "undefined")
    return null;
  return window.navigator.language || window.navigator.languages[0];
};
const monadicMemoize = (fn) => {
  const cache = /* @__PURE__ */ Object.create(null);
  const memoizedFn = (arg) => {
    const cacheKey = JSON.stringify(arg);
    if (cacheKey in cache) {
      return cache[cacheKey];
    }
    return cache[cacheKey] = fn(arg);
  };
  return memoizedFn;
};
var __defProp = Object.defineProperty;
var __getOwnPropSymbols = Object.getOwnPropertySymbols;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __propIsEnum = Object.prototype.propertyIsEnumerable;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __spreadValues = (a, b) => {
  for (var prop in b || (b = {}))
    if (__hasOwnProp.call(b, prop))
      __defNormalProp(a, prop, b[prop]);
  if (__getOwnPropSymbols)
    for (var prop of __getOwnPropSymbols(b)) {
      if (__propIsEnum.call(b, prop))
        __defNormalProp(a, prop, b[prop]);
    }
  return a;
};
var __objRest = (source, exclude) => {
  var target = {};
  for (var prop in source)
    if (__hasOwnProp.call(source, prop) && exclude.indexOf(prop) < 0)
      target[prop] = source[prop];
  if (source != null && __getOwnPropSymbols)
    for (var prop of __getOwnPropSymbols(source)) {
      if (exclude.indexOf(prop) < 0 && __propIsEnum.call(source, prop))
        target[prop] = source[prop];
    }
  return target;
};
const getIntlFormatterOptions = (type, name) => {
  const { formats } = getOptions();
  if (type in formats && name in formats[type]) {
    return formats[type][name];
  }
  throw new Error(`[svelte-i18n] Unknown "${name}" ${type} format.`);
};
const createNumberFormatter = monadicMemoize(
  (_a2) => {
    var _b = _a2, { locale, format } = _b, options2 = __objRest(_b, ["locale", "format"]);
    if (locale == null) {
      throw new Error('[svelte-i18n] A "locale" must be set to format numbers');
    }
    if (format) {
      options2 = getIntlFormatterOptions("number", format);
    }
    return new Intl.NumberFormat(locale, options2);
  }
);
const createDateFormatter = monadicMemoize(
  (_c) => {
    var _d = _c, { locale, format } = _d, options2 = __objRest(_d, ["locale", "format"]);
    if (locale == null) {
      throw new Error('[svelte-i18n] A "locale" must be set to format dates');
    }
    if (format) {
      options2 = getIntlFormatterOptions("date", format);
    } else if (Object.keys(options2).length === 0) {
      options2 = getIntlFormatterOptions("date", "short");
    }
    return new Intl.DateTimeFormat(locale, options2);
  }
);
const createTimeFormatter = monadicMemoize(
  (_e) => {
    var _f = _e, { locale, format } = _f, options2 = __objRest(_f, ["locale", "format"]);
    if (locale == null) {
      throw new Error(
        '[svelte-i18n] A "locale" must be set to format time values'
      );
    }
    if (format) {
      options2 = getIntlFormatterOptions("time", format);
    } else if (Object.keys(options2).length === 0) {
      options2 = getIntlFormatterOptions("time", "short");
    }
    return new Intl.DateTimeFormat(locale, options2);
  }
);
const getNumberFormatter = (_g = {}) => {
  var _h = _g, {
    locale = getCurrentLocale()
  } = _h, args = __objRest(_h, [
    "locale"
  ]);
  return createNumberFormatter(__spreadValues({ locale }, args));
};
const getDateFormatter = (_i = {}) => {
  var _j = _i, {
    locale = getCurrentLocale()
  } = _j, args = __objRest(_j, [
    "locale"
  ]);
  return createDateFormatter(__spreadValues({ locale }, args));
};
const getTimeFormatter = (_k = {}) => {
  var _l = _k, {
    locale = getCurrentLocale()
  } = _l, args = __objRest(_l, [
    "locale"
  ]);
  return createTimeFormatter(__spreadValues({ locale }, args));
};
const getMessageFormatter = monadicMemoize(
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  (message, locale = getCurrentLocale()) => new IntlMessageFormat(message, locale, getOptions().formats, {
    ignoreTag: getOptions().ignoreTag
  })
);
const formatMessage = (id, options2 = {}) => {
  var _a2, _b, _c, _d;
  let messageObj = options2;
  if (typeof id === "object") {
    messageObj = id;
    id = messageObj.id;
  }
  const {
    values,
    locale = getCurrentLocale(),
    default: defaultValue
  } = messageObj;
  if (locale == null) {
    throw new Error(
      "[svelte-i18n] Cannot format a message without first setting the initial locale."
    );
  }
  let message = lookup(id, locale);
  if (!message) {
    message = (_d = (_c = (_b = (_a2 = getOptions()).handleMissingMessage) == null ? void 0 : _b.call(_a2, { locale, id, defaultValue })) != null ? _c : defaultValue) != null ? _d : id;
  } else if (typeof message !== "string") {
    console.warn(
      `[svelte-i18n] Message with id "${id}" must be of type "string", found: "${typeof message}". Gettin its value through the "$format" method is deprecated; use the "json" method instead.`
    );
    return message;
  }
  if (!values) {
    return message;
  }
  let result = message;
  try {
    result = getMessageFormatter(message, locale).format(values);
  } catch (e) {
    if (e instanceof Error) {
      console.warn(
        `[svelte-i18n] Message "${id}" has syntax error:`,
        e.message
      );
    }
  }
  return result;
};
const formatTime = (t, options2) => {
  return getTimeFormatter(options2).format(t);
};
const formatDate = (d, options2) => {
  return getDateFormatter(options2).format(d);
};
const formatNumber = (n, options2) => {
  return getNumberFormatter(options2).format(n);
};
const getJSON = (id, locale = getCurrentLocale()) => {
  return lookup(id, locale);
};
const $format = derived([$locale, $dictionary], () => formatMessage);
derived([$locale], () => formatTime);
derived([$locale], () => formatDate);
derived([$locale], () => formatNumber);
derived([$locale, $dictionary], () => getJSON);
const navbar_config = writable(null);
let supports_adopted_stylesheets = false;
if (typeof window !== "undefined" && "attachShadow" in Element.prototype && "adoptedStyleSheets" in Document.prototype) {
  const shadow_root_test = document.createElement("div").attachShadow({ mode: "open" });
  supports_adopted_stylesheets = "adoptedStyleSheets" in shadow_root_test;
}
function mount_css(url, target) {
  const base = new URL(import.meta.url).origin;
  var _url = url;
  if (window.location.origin !== base) {
    _url = new URL(url, base).href;
  }
  const existing_link = document.querySelector(`link[href='${_url}']`);
  if (existing_link) return Promise.resolve();
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = _url;
  return new Promise((res, rej) => {
    link.addEventListener("load", () => res());
    link.addEventListener("error", () => {
      console.error(`Unable to preload CSS for ${_url}`);
      res();
    });
    target.appendChild(link);
  });
}
function prefix_css(string, version, style_element) {
  if (!supports_adopted_stylesheets) return string;
  if (!style_element) {
    style_element = document.createElement("style");
  }
  style_element.remove();
  const stylesheet = new CSSStyleSheet();
  stylesheet.replaceSync(string);
  let importString = "";
  string = string.replace(/@import\s+url\((.*?)\);\s*/g, (match, url) => {
    importString += `@import url(${url});
`;
    return "";
  });
  const rules = stylesheet.cssRules;
  let css_string = "";
  let gradio_css_infix = `.gradio-container.gradio-container-${version} .contain `;
  for (let i = 0; i < rules.length; i++) {
    const rule = rules[i];
    let is_dark_rule = rule.cssText.includes(".dark");
    if (rule instanceof CSSStyleRule) {
      const selector = rule.selectorText;
      if (selector) {
        const new_selector = selector.replace(".dark", "").split(",").map(
          (s) => `${is_dark_rule ? ".dark" : ""} ${gradio_css_infix} ${s.trim()} `
        ).join(",");
        css_string += rule.cssText;
        css_string += rule.cssText.replace(selector, new_selector);
      }
    } else if (rule instanceof CSSMediaRule) {
      let mediaCssString = `@media ${rule.media.mediaText} {`;
      for (let j = 0; j < rule.cssRules.length; j++) {
        const innerRule = rule.cssRules[j];
        if (innerRule instanceof CSSStyleRule) {
          let is_dark_rule2 = innerRule.cssText.includes(".dark ");
          const selector = innerRule.selectorText;
          const new_selector = selector.replace(".dark", "").split(",").map(
            (s) => `${is_dark_rule2 ? ".dark" : ""} ${gradio_css_infix} ${s.trim()} `
          ).join(",");
          mediaCssString += innerRule.cssText.replace(selector, new_selector);
        }
      }
      mediaCssString += "}";
      css_string += mediaCssString;
    } else if (rule instanceof CSSKeyframesRule) {
      css_string += `@keyframes ${rule.name} {`;
      for (let j = 0; j < rule.cssRules.length; j++) {
        const innerRule = rule.cssRules[j];
        if (innerRule instanceof CSSKeyframeRule) {
          css_string += `${innerRule.keyText} { ${innerRule.style.cssText} }`;
        }
      }
      css_string += "}";
    } else if (rule instanceof CSSFontFaceRule) {
      css_string += `@font-face { ${rule.style.cssText} }`;
    }
  }
  return importString + css_string;
}
const component_map = {
  "accordion": {
    component: () => import('./Index32-Bne_RZGi.js')
  },
  "annotatedimage": {
    component: () => import('./Index19-lTckZTfY.js')
  },
  "audio": {
    base: () => import('./StaticAudio-CGENs1q8.js').then((n) => n.a),
    example: () => import('./Example10-DPWDxkA2.js'),
    component: () => import('./index53-37ifDFpM.js')
  },
  "box": {
    component: () => import('./Index6-BXTaq0C-.js')
  },
  "browserstate": {
    component: () => import('./Index56-CwrnHgQO.js')
  },
  "button": {
    component: () => import('./Index7-D46XnVSh.js')
  },
  "chatbot": {
    component: () => import('./Index54-DLt_GW8d.js')
  },
  "checkbox": {
    example: () => import('./Example11-CiOo3Xzc.js'),
    component: () => import('./Index8-BFkZJO40.js')
  },
  "checkboxgroup": {
    example: () => import('./Example12-C3QB1uOZ.js'),
    component: () => import('./Index20-B0YukPrc.js')
  },
  "code": {
    example: () => import('./Example-DQafirJP.js'),
    component: () => import('./Index52-Co7gLzPI.js')
  },
  "colorpicker": {
    example: () => import('./Example13-8uwQHwRN.js'),
    component: () => import('./Index40-yihVSKCF.js')
  },
  "column": {
    component: () => import('./Index9-Bh3w6OXH.js')
  },
  "core": {
    component: () => import('./index-DuBh77GF.js')
  },
  "dataframe": {
    example: () => import('./Example14-B9QZm81N.js'),
    component: () => import('./Index55-sLnjb71O.js')
  },
  "dataset": {
    component: () => import('./Index44-aSPrr52Q.js')
  },
  "datetime": {
    example: () => import('./Example3-CxHLke4-.js'),
    component: () => import('./Index45-CbtIq1Mh.js')
  },
  "dialogue": {
    example: () => import('./Example15-Xubv-cAE.js'),
    component: () => import('./Index46-Ba88O_r1.js')
  },
  "downloadbutton": {
    component: () => import('./Index33-BNqtueh_.js')
  },
  "draggable": {
    component: () => import('./Index21-C0LViUOE.js')
  },
  "dropdown": {
    example: () => import('./Example4-CMk5LZQq.js'),
    component: () => import('./Index41-P4de9cQY.js')
  },
  "file": {
    example: () => import('./Example5-DyYnewTx.js'),
    component: () => import('./Index22-DAGaNe0Y.js')
  },
  "fileexplorer": {
    example: () => import('./Example16-Dao3Xx9_.js'),
    component: () => import('./Index51-v825UUE6.js')
  },
  "form": {
    component: () => import('./Index10-CXU7WwXL.js')
  },
  "gallery": {
    base: () => import('./Gallery-CTR9kkPc.js').then((n) => n.a),
    example: () => import('./Example17-eUqd0Yc_.js'),
    component: () => import('./Index23-DCUEAM82.js')
  },
  "group": {
    component: () => import('./Index24-BJnnbNeH.js')
  },
  "highlightedtext": {
    component: () => import('./Index48-DrPj8942.js')
  },
  "html": {
    base: () => import('./HTML-C43qA6n7.js'),
    example: () => import('./Example18-BZiH_GVV.js'),
    component: () => import('./Index25-C7ulWp_5.js')
  },
  "image": {
    base: () => import('./ImagePreview-CjXAuu8E.js'),
    example: () => import('./Example6-CY1BLAOH.js'),
    component: () => import('./Index26-CFj0AWf4.js')
  },
  "imageeditor": {
    example: () => import('./Example19-Co3rLOUG.js'),
    component: () => import('./Index4-DjOJEvOl.js').then((n) => n.au)
  },
  "imageslider": {
    example: () => import('./Example20-D6mdG3QW.js'),
    component: () => import('./Index57-D9780Wsu.js')
  },
  "json": {
    example: () => import('./Example21-CHSL5Ikg.js'),
    component: () => import('./Index11-Bi_xLLav.js')
  },
  "label": {
    component: () => import('./Index34-D97S0Jn5.js')
  },
  "markdown": {
    example: () => import('./Example7-CLPLiEhz.js'),
    component: () => import('./Index12-M-dcVqUC.js')
  },
  "model3d": {
    base: () => import('./Model3D-Byrr9Lao.js'),
    example: () => import('./Example22-Bx6IyqEy.js'),
    component: () => import('./Index35-C85cD6o-.js')
  },
  "multimodaltextbox": {
    example: () => import('./Example23-Cdj-Vjy_.js'),
    component: () => import('./Index47-DYRKVxF0.js')
  },
  "nativeplot": {
    example: () => import('./Example8-CwW5sBgQ.js'),
    component: () => import('./Index27-G5VczEpy.js')
  },
  "navbar": {
    base: () => import('./Index13-TRAfPNAM.js'),
    example: () => import('./Example9-BJqhfojB.js'),
    component: () => import('./Index13-TRAfPNAM.js')
  },
  "number": {
    example: () => import('./Example24-DZseNWl5.js'),
    component: () => import('./Index28-BmqNkw5i.js')
  },
  "paramviewer": {
    example: () => import('./Example25-ArigFhpQ.js'),
    component: () => import('./Index49-BZTDsGNu.js')
  },
  "plot": {
    base: () => import('./Plot-DQRGUyXc.js').then((n) => n.b),
    component: () => import('./Index14-C-Lz1a5W.js')
  },
  "radio": {
    example: () => import('./Example26-4aUdI3US.js'),
    component: () => import('./Index42-Zgwdd5yx.js')
  },
  "row": {
    component: () => import('./Index29-RpB1Yxk5.js')
  },
  "sidebar": {
    component: () => import('./Index36-xwuDeO_Q.js')
  },
  "slider": {
    example: () => import('./Example27-DeavLB6j.js'),
    component: () => import('./Index30-CoESKgx2.js')
  },
  "state": {
    component: () => import('./Index15-RFWCUaoO.js')
  },
  "statustracker": {
    component: () => import('./index2-DQ0rufig.js')
  },
  "tabitem": {
    component: () => import('./Index37-Bbib4BAM.js')
  },
  "tabs": {
    component: () => import('./Index31-X-tcmTYk.js')
  },
  "textbox": {
    example: () => import('./Example2-8iT2brQo.js'),
    component: () => import('./Index16-BmyX54vt.js')
  },
  "timer": {
    component: () => import('./Index17-B91_ri0E.js')
  },
  "uploadbutton": {
    component: () => import('./Index38-CNGW8M28.js')
  },
  "vibeeditor": {
    component: () => import('./Index18-DNsben2n.js')
  },
  "video": {
    base: () => import('./VideoPreview-BUKiZR1J.js').then((n) => n.b),
    example: () => import('./Example28-D1-aV4K8.js'),
    component: () => import('./index43-DMLSOqb-.js')
  },
  "walkthrough": {
    component: () => import('./Index31-X-tcmTYk.js')
  },
  "walkthroughstep": {
    component: () => import('./Index37-Bbib4BAM.js')
  }
};
const request_map = {};
const runtime_map = {};
const is_browser = typeof window !== "undefined";
function load_component({ api_url, name, id, variant }) {
  const comps = is_browser && window.__GRADIO__CC__;
  const _component_map = {
    // eslint-disable-next-line no-undef
    ...component_map,
    ...!comps ? {} : comps
  };
  let _id = id || name;
  if (request_map[`${_id}-${variant}`]) {
    return {
      component: request_map[`${_id}-${variant}`],
      name,
      runtime: runtime_map[`${_id}-${variant}`]
    };
  }
  try {
    if (!_component_map?.[_id]?.[variant] && !_component_map?.[name]?.[variant])
      throw new Error();
    request_map[`${_id}-${variant}`] = (_component_map?.[_id]?.[variant] || // for dev mode custom components
    _component_map?.[name]?.[variant])();
    runtime_map[`${_id}-${variant}`] = is_browser && window.__GRADIO__CC__RUNTIMES__?.[id] || // for dev mode custom components
    false;
    return {
      name,
      component: request_map[`${_id}-${variant}`],
      runtime: runtime_map[`${_id}-${variant}`]
    };
  } catch (e) {
    if (!_id) throw new Error(`Component not found: ${name}`);
    try {
      const cc = get_component_with_css(api_url, _id, variant);
      const [component_module, svelte_runtime_module] = cc;
      request_map[`${_id}-${variant}`] = component_module;
      runtime_map[`${_id}-${variant}`] = svelte_runtime_module;
      return {
        name,
        component: request_map[`${_id}-${variant}`],
        runtime: runtime_map[`${_id}-${variant}`]
      };
    } catch (e2) {
      if (variant === "example") {
        request_map[`${_id}-${variant}`] = import('./Example29-CaLCf-6A.js');
        return {
          name,
          component: request_map[`${_id}-${variant}`],
          runtime: runtime_map[`${_id}-${variant}`]
        };
      }
      console.error(`failed to load: ${name}`);
      console.error(e2);
      throw e2;
    }
  }
}
function load_css(url) {
  if (!is_browser) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = url;
    document.head.appendChild(link);
    link.onload = () => resolve();
    link.onerror = () => reject();
  });
}
function get_component_with_css(api_url, id, variant) {
  const environment = is_browser ? "client" : "server";
  if (environment === "server") {
    return [import('./Index39-C-nV5fyj.js'), Promise.resolve(false)];
  }
  const path = `${api_url}/custom_component/${id}/${environment}/${variant}/index.js`;
  return [
    Promise.all([
      load_css(
        `${api_url}/custom_component/${id}/${environment}/${variant}/style.css`
      ),
      import(
        /* @vite-ignore */
        path
      )
    ]).then(([_, component_module]) => {
      return component_module;
    }),
    import(
      /* @vite-ignore */
      `${api_url}/custom_component/${id}/${environment}/${variant}/svelte_runtime_entry.js`
    )
  ];
}
function get_component(type, class_id, root, variant = "component") {
  if (type === "api") type = "state";
  return load_component({
    api_url: root,
    name: type,
    id: class_id,
    variant
  });
}
function get_inputs_outputs(dependencies) {
  const inputs = /* @__PURE__ */ new Set();
  const outputs = /* @__PURE__ */ new Set();
  for (const dep of dependencies) {
    dep.inputs.forEach((input) => inputs.add(input));
    dep.outputs.forEach((output) => outputs.add(output));
  }
  return [inputs, outputs];
}
function determine_interactivity(id, interactive_prop, value, dependencies) {
  const [inputs, outputs] = dependencies;
  if (interactive_prop === false) {
    return false;
  } else if (interactive_prop === true) {
    return true;
  } else if (inputs.has(id) || !outputs.has(id) && has_no_default_value(value)) {
    return true;
  }
  return false;
}
function has_no_default_value(value) {
  return Array.isArray(value) && value.length === 0 || value === "" || value === 0 || !value;
}
const AsyncFunction = Object.getPrototypeOf(
  async function() {
  }
).constructor;
const color_values = [
  { color: "red", primary: 600, secondary: 100 },
  { color: "green", primary: 600, secondary: 100 },
  { color: "blue", primary: 600, secondary: 100 },
  { color: "yellow", primary: 500, secondary: 100 },
  { color: "purple", primary: 600, secondary: 100 },
  { color: "teal", primary: 600, secondary: 100 },
  { color: "orange", primary: 600, secondary: 100 },
  { color: "cyan", primary: 600, secondary: 100 },
  { color: "lime", primary: 500, secondary: 100 },
  { color: "pink", primary: 600, secondary: 100 }
];
const tw_colors = {
  inherit: "inherit",
  current: "currentColor",
  transparent: "transparent",
  black: "#000",
  white: "#fff",
  slate: {
    50: "#f8fafc",
    100: "#f1f5f9",
    200: "#e2e8f0",
    300: "#cbd5e1",
    400: "#94a3b8",
    500: "#64748b",
    600: "#475569",
    700: "#334155",
    800: "#1e293b",
    900: "#0f172a",
    950: "#020617"
  },
  gray: {
    50: "#f9fafb",
    100: "#f3f4f6",
    200: "#e5e7eb",
    300: "#d1d5db",
    400: "#9ca3af",
    500: "#6b7280",
    600: "#4b5563",
    700: "#374151",
    800: "#1f2937",
    900: "#111827",
    950: "#030712"
  },
  zinc: {
    50: "#fafafa",
    100: "#f4f4f5",
    200: "#e4e4e7",
    300: "#d4d4d8",
    400: "#a1a1aa",
    500: "#71717a",
    600: "#52525b",
    700: "#3f3f46",
    800: "#27272a",
    900: "#18181b",
    950: "#09090b"
  },
  neutral: {
    50: "#fafafa",
    100: "#f5f5f5",
    200: "#e5e5e5",
    300: "#d4d4d4",
    400: "#a3a3a3",
    500: "#737373",
    600: "#525252",
    700: "#404040",
    800: "#262626",
    900: "#171717",
    950: "#0a0a0a"
  },
  stone: {
    50: "#fafaf9",
    100: "#f5f5f4",
    200: "#e7e5e4",
    300: "#d6d3d1",
    400: "#a8a29e",
    500: "#78716c",
    600: "#57534e",
    700: "#44403c",
    800: "#292524",
    900: "#1c1917",
    950: "#0c0a09"
  },
  red: {
    50: "#fef2f2",
    100: "#fee2e2",
    200: "#fecaca",
    300: "#fca5a5",
    400: "#f87171",
    500: "#ef4444",
    600: "#dc2626",
    700: "#b91c1c",
    800: "#991b1b",
    900: "#7f1d1d",
    950: "#450a0a"
  },
  orange: {
    50: "#fff7ed",
    100: "#ffedd5",
    200: "#fed7aa",
    300: "#fdba74",
    400: "#fb923c",
    500: "#f97316",
    600: "#ea580c",
    700: "#c2410c",
    800: "#9a3412",
    900: "#7c2d12",
    950: "#431407"
  },
  amber: {
    50: "#fffbeb",
    100: "#fef3c7",
    200: "#fde68a",
    300: "#fcd34d",
    400: "#fbbf24",
    500: "#f59e0b",
    600: "#d97706",
    700: "#b45309",
    800: "#92400e",
    900: "#78350f",
    950: "#451a03"
  },
  yellow: {
    50: "#fefce8",
    100: "#fef9c3",
    200: "#fef08a",
    300: "#fde047",
    400: "#facc15",
    500: "#eab308",
    600: "#ca8a04",
    700: "#a16207",
    800: "#854d0e",
    900: "#713f12",
    950: "#422006"
  },
  lime: {
    50: "#f7fee7",
    100: "#ecfccb",
    200: "#d9f99d",
    300: "#bef264",
    400: "#a3e635",
    500: "#84cc16",
    600: "#65a30d",
    700: "#4d7c0f",
    800: "#3f6212",
    900: "#365314",
    950: "#1a2e05"
  },
  green: {
    50: "#f0fdf4",
    100: "#dcfce7",
    200: "#bbf7d0",
    300: "#86efac",
    400: "#4ade80",
    500: "#22c55e",
    600: "#16a34a",
    700: "#15803d",
    800: "#166534",
    900: "#14532d",
    950: "#052e16"
  },
  emerald: {
    50: "#ecfdf5",
    100: "#d1fae5",
    200: "#a7f3d0",
    300: "#6ee7b7",
    400: "#34d399",
    500: "#10b981",
    600: "#059669",
    700: "#047857",
    800: "#065f46",
    900: "#064e3b",
    950: "#022c22"
  },
  teal: {
    50: "#f0fdfa",
    100: "#ccfbf1",
    200: "#99f6e4",
    300: "#5eead4",
    400: "#2dd4bf",
    500: "#14b8a6",
    600: "#0d9488",
    700: "#0f766e",
    800: "#115e59",
    900: "#134e4a",
    950: "#042f2e"
  },
  cyan: {
    50: "#ecfeff",
    100: "#cffafe",
    200: "#a5f3fc",
    300: "#67e8f9",
    400: "#22d3ee",
    500: "#06b6d4",
    600: "#0891b2",
    700: "#0e7490",
    800: "#155e75",
    900: "#164e63",
    950: "#083344"
  },
  sky: {
    50: "#f0f9ff",
    100: "#e0f2fe",
    200: "#bae6fd",
    300: "#7dd3fc",
    400: "#38bdf8",
    500: "#0ea5e9",
    600: "#0284c7",
    700: "#0369a1",
    800: "#075985",
    900: "#0c4a6e",
    950: "#082f49"
  },
  blue: {
    50: "#eff6ff",
    100: "#dbeafe",
    200: "#bfdbfe",
    300: "#93c5fd",
    400: "#60a5fa",
    500: "#3b82f6",
    600: "#2563eb",
    700: "#1d4ed8",
    800: "#1e40af",
    900: "#1e3a8a",
    950: "#172554"
  },
  indigo: {
    50: "#eef2ff",
    100: "#e0e7ff",
    200: "#c7d2fe",
    300: "#a5b4fc",
    400: "#818cf8",
    500: "#6366f1",
    600: "#4f46e5",
    700: "#4338ca",
    800: "#3730a3",
    900: "#312e81",
    950: "#1e1b4b"
  },
  violet: {
    50: "#f5f3ff",
    100: "#ede9fe",
    200: "#ddd6fe",
    300: "#c4b5fd",
    400: "#a78bfa",
    500: "#8b5cf6",
    600: "#7c3aed",
    700: "#6d28d9",
    800: "#5b21b6",
    900: "#4c1d95",
    950: "#2e1065"
  },
  purple: {
    50: "#faf5ff",
    100: "#f3e8ff",
    200: "#e9d5ff",
    300: "#d8b4fe",
    400: "#c084fc",
    500: "#a855f7",
    600: "#9333ea",
    700: "#7e22ce",
    800: "#6b21a8",
    900: "#581c87",
    950: "#3b0764"
  },
  fuchsia: {
    50: "#fdf4ff",
    100: "#fae8ff",
    200: "#f5d0fe",
    300: "#f0abfc",
    400: "#e879f9",
    500: "#d946ef",
    600: "#c026d3",
    700: "#a21caf",
    800: "#86198f",
    900: "#701a75",
    950: "#4a044e"
  },
  pink: {
    50: "#fdf2f8",
    100: "#fce7f3",
    200: "#fbcfe8",
    300: "#f9a8d4",
    400: "#f472b6",
    500: "#ec4899",
    600: "#db2777",
    700: "#be185d",
    800: "#9d174d",
    900: "#831843",
    950: "#500724"
  },
  rose: {
    50: "#fff1f2",
    100: "#ffe4e6",
    200: "#fecdd3",
    300: "#fda4af",
    400: "#fb7185",
    500: "#f43f5e",
    600: "#e11d48",
    700: "#be123c",
    800: "#9f1239",
    900: "#881337",
    950: "#4c0519"
  }
};
color_values.reduce(
  (acc, { color, primary, secondary }) => ({
    ...acc,
    [color]: {
      primary: tw_colors[color][primary],
      secondary: tw_colors[color][secondary]
    }
  }),
  {}
);
const HOST_URL = `host`;
const SSE_URL = `queue/data`;
const SSE_DATA_URL = `queue/join`;
const UPLOAD_URL = `upload`;
const LOGIN_URL = `login`;
const CONFIG_URL = `config`;
const API_INFO_URL = `info`;
const RUNTIME_URL = `runtime`;
const SLEEPTIME_URL = `sleeptime`;
const HEARTBEAT_URL = `heartbeat`;
const COMPONENT_SERVER_URL = `component_server`;
const RESET_URL = `reset`;
const CANCEL_URL = `cancel`;
const APP_ID_URL = `app_id`;
const QUEUE_FULL_MSG = "This application is currently busy. Please try again. ";
const BROKEN_CONNECTION_MSG = "Connection errored out. ";
const CONFIG_ERROR_MSG = "Could not resolve app config. ";
const SPACE_STATUS_ERROR_MSG = "Could not get space status. ";
const API_INFO_ERROR_MSG = "Could not get API info. ";
const SPACE_METADATA_ERROR_MSG = "Space metadata could not be loaded. ";
const INVALID_URL_MSG = "Invalid URL. A full URL path is required.";
const UNAUTHORIZED_MSG = "Not authorized to access this space. ";
const INVALID_CREDENTIALS_MSG = "Invalid credentials. Could not login. ";
const MISSING_CREDENTIALS_MSG = "Login credentials are required to access this space.";
const NODEJS_FS_ERROR_MSG = "File system access is only available in Node.js environments";
const ROOT_URL_ERROR_MSG = "Root URL not found in client config";
const FILE_PROCESSING_ERROR_MSG = "Error uploading file";
async function get_jwt(space, token, cookies) {
  try {
    const r = await fetch(`https://huggingface.co/api/spaces/${space}/jwt`, {
      headers: {
        Authorization: `Bearer ${token}`,
        ...cookies ? { Cookie: cookies } : {}
      }
    });
    const jwt = (await r.json()).token;
    return jwt || false;
  } catch (e) {
    return false;
  }
}
function map_names_to_ids(fns) {
  let apis = {};
  fns.forEach(({ api_name, id }) => {
    if (api_name) apis[api_name] = id;
  });
  return apis;
}
async function resolve_config(endpoint) {
  const headers = this.options.token ? { Authorization: `Bearer ${this.options.token}` } : {};
  headers["Content-Type"] = "application/json";
  if (typeof window !== "undefined" && window.gradio_config && location.origin !== "http://localhost:9876") {
    if (window.gradio_config.current_page) {
      endpoint = endpoint.substring(0, endpoint.lastIndexOf("/"));
    }
    if (window.gradio_config.dev_mode || typeof window !== "undefined" && window?.BUILD_MODE === "dev") {
      let config_url = join_urls(
        endpoint,
        this.deep_link ? CONFIG_URL + "?deep_link=" + this.deep_link : CONFIG_URL
      );
      const response = await this.fetch(config_url, {
        headers,
        credentials: "include"
      });
      const config = await handleConfigResponse(response, !!this.options.auth);
      config.root = endpoint || config.root;
      window.gradio_config = {
        ...config,
        current_page: window.gradio_config.current_page
      };
    }
    return { ...window.gradio_config };
  } else if (endpoint) {
    let config_url = join_urls(
      endpoint,
      this.deep_link ? CONFIG_URL + "?deep_link=" + this.deep_link : CONFIG_URL
    );
    const response = await this.fetch(config_url, {
      headers,
      credentials: "include"
    });
    const config = await handleConfigResponse(response, !!this.options.auth);
    if (!config.root) {
      config.root = endpoint;
    }
    return config;
  }
  throw new Error(CONFIG_ERROR_MSG);
}
async function handleConfigResponse(response, authorized) {
  if (response?.status === 401 && !authorized) {
    const error_data = await response.json();
    const auth_message = error_data?.detail?.auth_message;
    throw new Error(auth_message || MISSING_CREDENTIALS_MSG);
  } else if (response?.status === 401 && authorized) {
    throw new Error(INVALID_CREDENTIALS_MSG);
  }
  if (response?.status === 200) {
    let config = await response.json();
    config.dependencies?.forEach((dep, i) => {
      if (dep.id === void 0) {
        dep.id = i;
      }
    });
    return config;
  } else if (response?.status === 401) {
    throw new Error(UNAUTHORIZED_MSG);
  }
  throw new Error(CONFIG_ERROR_MSG);
}
async function resolve_cookies() {
  const { http_protocol, host } = await process_endpoint(
    this.app_reference,
    this.options.token
  );
  try {
    if (this.options.auth) {
      const cookie_header = await get_cookie_header(
        http_protocol,
        host,
        this.options.auth,
        this.fetch,
        this.options.token
      );
      if (cookie_header) this.set_cookies(cookie_header);
    }
  } catch (e) {
    throw Error(e.message);
  }
}
async function get_cookie_header(http_protocol, host, auth, _fetch, token) {
  const formData = new FormData();
  formData.append("username", auth?.[0]);
  formData.append("password", auth?.[1]);
  let headers = {};
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  const res = await _fetch(`${http_protocol}//${host}/${LOGIN_URL}`, {
    headers,
    method: "POST",
    body: formData,
    credentials: "include"
  });
  if (res.status === 200) {
    return res.headers.get("set-cookie");
  } else if (res.status === 401) {
    throw new Error(INVALID_CREDENTIALS_MSG);
  } else {
    throw new Error(SPACE_METADATA_ERROR_MSG);
  }
}
function determine_protocol(endpoint) {
  if (endpoint.startsWith("http")) {
    const { protocol, host, pathname } = new URL(endpoint);
    return {
      ws_protocol: protocol === "https:" ? "wss" : "ws",
      http_protocol: protocol,
      host: host + (pathname !== "/" ? pathname : "")
    };
  }
  return {
    ws_protocol: "wss",
    http_protocol: "https:",
    host: new URL(endpoint).host
  };
}
const parse_and_set_cookies = (cookie_header) => {
  let cookies = [];
  const parts = cookie_header.split(/,(?=\s*[^\s=;]+=[^\s=;]+)/);
  parts.forEach((cookie) => {
    const [cookie_name, cookie_value] = cookie.split(";")[0].split("=");
    if (cookie_name && cookie_value) {
      cookies.push(`${cookie_name.trim()}=${cookie_value.trim()}`);
    }
  });
  return cookies;
};
const RE_SPACE_NAME = /^[a-zA-Z0-9_\-\.]+\/[a-zA-Z0-9_\-\.]+$/;
const RE_SPACE_DOMAIN = /.*hf\.space\/{0,1}.*$/;
async function process_endpoint(app_reference, token) {
  const headers = {};
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  const _app_reference = app_reference.trim().replace(/\/$/, "");
  if (RE_SPACE_NAME.test(_app_reference)) {
    try {
      const res = await fetch(
        `https://huggingface.co/api/spaces/${_app_reference}/${HOST_URL}`,
        { headers }
      );
      const _host = (await res.json()).host;
      return {
        space_id: app_reference,
        ...determine_protocol(_host)
      };
    } catch (e) {
      throw new Error(SPACE_METADATA_ERROR_MSG);
    }
  }
  if (RE_SPACE_DOMAIN.test(_app_reference)) {
    const { ws_protocol, http_protocol, host } = determine_protocol(_app_reference);
    return {
      space_id: host.split("/")[0].replace(".hf.space", ""),
      ws_protocol,
      http_protocol,
      host
    };
  }
  return {
    space_id: false,
    ...determine_protocol(_app_reference)
  };
}
const join_urls = (...urls) => {
  try {
    return urls.reduce((base_url, part) => {
      base_url = base_url.replace(/\/+$/, "");
      part = part.replace(/^\/+/, "");
      return new URL(part, base_url + "/").toString();
    });
  } catch (e) {
    throw new Error(INVALID_URL_MSG);
  }
};
function transform_api_info(api_info, config, api_map) {
  const transformed_info = {
    named_endpoints: {},
    unnamed_endpoints: {}
  };
  Object.keys(api_info).forEach((category) => {
    if (category === "named_endpoints" || category === "unnamed_endpoints") {
      transformed_info[category] = {};
      Object.entries(api_info[category]).forEach(
        ([endpoint, { parameters, returns }]) => {
          const dependencyIndex = config.dependencies.find(
            (dep) => dep.api_name === endpoint || dep.api_name === endpoint.replace("/", "")
          )?.id || api_map[endpoint.replace("/", "")] || -1;
          const dependencyTypes = dependencyIndex !== -1 ? config.dependencies.find((dep) => dep.id == dependencyIndex)?.types : { generator: false, cancel: false };
          if (dependencyIndex !== -1 && config.dependencies.find((dep) => dep.id == dependencyIndex)?.inputs?.length !== parameters.length) {
            const components = config.dependencies.find((dep) => dep.id == dependencyIndex).inputs.map(
              (input) => config.components.find((c) => c.id === input)?.type
            );
            try {
              components.forEach((comp, idx) => {
                if (comp === "state") {
                  const new_param = {
                    component: "state",
                    example: null,
                    parameter_default: null,
                    parameter_has_default: true,
                    parameter_name: null,
                    hidden: true
                  };
                  parameters.splice(idx, 0, new_param);
                }
              });
            } catch (e) {
              console.error(e);
            }
          }
          const transform_type = (data, component, serializer, signature_type) => ({
            ...data,
            description: get_description(data?.type, serializer),
            type: get_type(data?.type, component, serializer, signature_type) || ""
          });
          transformed_info[category][endpoint] = {
            parameters: parameters.map(
              (p) => transform_type(p, p?.component, p?.serializer, "parameter")
            ),
            returns: returns.map(
              (r) => transform_type(r, r?.component, r?.serializer, "return")
            ),
            type: dependencyTypes
          };
        }
      );
    }
  });
  return transformed_info;
}
function get_type(type, component, serializer, signature_type) {
  if (component === "Api") return type.type;
  switch (type?.type) {
    case "string":
      return "string";
    case "boolean":
      return "boolean";
    case "number":
      return "number";
  }
  if (serializer === "JSONSerializable" || serializer === "StringSerializable") {
    return "any";
  } else if (serializer === "ListStringSerializable") {
    return "string[]";
  } else if (component === "Image") {
    return signature_type === "parameter" ? "Blob | File | Buffer" : "string";
  } else if (serializer === "FileSerializable") {
    if (type?.type === "array") {
      return signature_type === "parameter" ? "(Blob | File | Buffer)[]" : `{ name: string; data: string; size?: number; is_file?: boolean; orig_name?: string}[]`;
    }
    return signature_type === "parameter" ? "Blob | File | Buffer" : `{ name: string; data: string; size?: number; is_file?: boolean; orig_name?: string}`;
  } else if (serializer === "GallerySerializable") {
    return signature_type === "parameter" ? "[(Blob | File | Buffer), (string | null)][]" : `[{ name: string; data: string; size?: number; is_file?: boolean; orig_name?: string}, (string | null))][]`;
  }
}
function get_description(type, serializer) {
  if (serializer === "GallerySerializable") {
    return "array of [file, label] tuples";
  } else if (serializer === "ListStringSerializable") {
    return "array of strings";
  } else if (serializer === "FileSerializable") {
    return "array of files or single file";
  }
  return type?.description;
}
function handle_message(data, last_status) {
  const queue2 = true;
  switch (data.msg) {
    case "send_data":
      return { type: "data" };
    case "send_hash":
      return { type: "hash" };
    case "queue_full":
      return {
        type: "update",
        status: {
          queue: queue2,
          message: QUEUE_FULL_MSG,
          stage: "error",
          code: data.code,
          success: data.success
        }
      };
    case "heartbeat":
      return {
        type: "heartbeat"
      };
    case "unexpected_error":
      return {
        type: "unexpected_error",
        status: {
          queue: queue2,
          message: data.message,
          session_not_found: data.session_not_found,
          stage: "error",
          success: false
        }
      };
    case "broken_connection":
      return {
        type: "broken_connection",
        status: {
          queue: queue2,
          message: data.message,
          stage: "error",
          success: false
        }
      };
    case "estimation":
      return {
        type: "update",
        status: {
          queue: queue2,
          stage: last_status || "pending",
          code: data.code,
          size: data.queue_size,
          position: data.rank,
          eta: data.rank_eta,
          success: data.success
        }
      };
    case "progress":
      return {
        type: "update",
        status: {
          queue: queue2,
          stage: "pending",
          code: data.code,
          progress_data: data.progress_data,
          success: data.success
        }
      };
    case "log":
      return { type: "log", data };
    case "process_generating":
      return {
        type: "generating",
        status: {
          queue: queue2,
          message: !data.success ? data.output.error : null,
          stage: data.success ? "generating" : "error",
          code: data.code,
          progress_data: data.progress_data,
          eta: data.average_duration,
          changed_state_ids: data.success ? data.output.changed_state_ids : void 0
        },
        data: data.success ? data.output : null
      };
    case "process_streaming":
      return {
        type: "streaming",
        status: {
          queue: queue2,
          message: data.output.error,
          stage: "streaming",
          time_limit: data.time_limit,
          code: data.code,
          progress_data: data.progress_data,
          eta: data.eta
        },
        data: data.output
      };
    case "process_completed":
      if ("error" in data.output) {
        return {
          type: "update",
          status: {
            queue: queue2,
            title: data.output.title ?? "Error",
            message: data.output.error ?? "An error occurred",
            visible: data.output.visible,
            duration: data.output.duration,
            stage: "error",
            code: data.code,
            success: data.success
          }
        };
      }
      return {
        type: "complete",
        status: {
          queue: queue2,
          message: !data.success ? data.output.error : void 0,
          stage: data.success ? "complete" : "error",
          code: data.code,
          progress_data: data.progress_data,
          changed_state_ids: data.success ? data.output.changed_state_ids : void 0
        },
        data: data.success ? data.output : null
      };
    case "process_starts":
      return {
        type: "update",
        status: {
          queue: queue2,
          stage: "pending",
          code: data.code,
          size: data.rank,
          position: 0,
          success: data.success,
          eta: data.eta
        },
        original_msg: "process_starts"
      };
  }
  return { type: "none", status: { stage: "error", queue: queue2 } };
}
const map_data_to_params = (data = [], endpoint_info) => {
  const parameters = endpoint_info ? endpoint_info.parameters : [];
  if (Array.isArray(data)) {
    if (endpoint_info && parameters.length > 0 && data.length > parameters.length) {
      console.warn("Too many arguments provided for the endpoint.");
    }
    return data;
  }
  const resolved_data = [];
  const provided_keys = Object.keys(data);
  parameters.forEach((param, index) => {
    if (data.hasOwnProperty(param.parameter_name)) {
      resolved_data[index] = data[param.parameter_name];
    } else if (param.parameter_has_default) {
      resolved_data[index] = param.parameter_default;
    } else {
      throw new Error(
        `No value provided for required parameter: ${param.parameter_name}`
      );
    }
  });
  provided_keys.forEach((key) => {
    if (!parameters.some((param) => param.parameter_name === key)) {
      throw new Error(
        `Parameter \`${key}\` is not a valid keyword argument. Please refer to the API for usage.`
      );
    }
  });
  resolved_data.forEach((value, idx) => {
    if (value === void 0 && !parameters[idx].parameter_has_default) {
      throw new Error(
        `No value provided for required parameter: ${parameters[idx].parameter_name}`
      );
    }
  });
  return resolved_data;
};
async function view_api() {
  if (this.api_info) return this.api_info;
  const { token } = this.options;
  const { config } = this;
  const headers = { "Content-Type": "application/json" };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  if (!config) {
    return;
  }
  try {
    let response;
    let api_info;
    if (typeof window !== "undefined" && window.gradio_api_info) {
      api_info = window.gradio_api_info;
    } else {
      const url = join_urls(config.root, this.api_prefix, API_INFO_URL);
      response = await this.fetch(url, {
        headers,
        credentials: "include"
      });
      if (!response.ok) {
        throw new Error(BROKEN_CONNECTION_MSG);
      }
      api_info = await response.json();
    }
    if ("api" in api_info) {
      api_info = api_info.api;
    }
    if (api_info.named_endpoints["/predict"] && !api_info.unnamed_endpoints["0"]) {
      api_info.unnamed_endpoints[0] = api_info.named_endpoints["/predict"];
    }
    return transform_api_info(api_info, config, this.api_map);
  } catch (e) {
    throw new Error("Could not get API info. " + e.message);
  }
}
async function upload_files(root_url, files, upload_id) {
  const headers = {};
  if (this?.options?.token) {
    headers.Authorization = `Bearer ${this.options.token}`;
  }
  const chunkSize = 1e3;
  const uploadResponses = [];
  let response;
  for (let i = 0; i < files.length; i += chunkSize) {
    const chunk = files.slice(i, i + chunkSize);
    const formData = new FormData();
    chunk.forEach((file2) => {
      formData.append("files", file2);
    });
    try {
      const upload_url = upload_id ? `${root_url}${this.api_prefix}/${UPLOAD_URL}?upload_id=${upload_id}` : `${root_url}${this.api_prefix}/${UPLOAD_URL}`;
      response = await this.fetch(upload_url, {
        method: "POST",
        body: formData,
        headers,
        credentials: "include"
      });
    } catch (e) {
      throw new Error(BROKEN_CONNECTION_MSG + e.message);
    }
    if (!response.ok) {
      const error_text = await response.text();
      return { error: `HTTP ${response.status}: ${error_text}` };
    }
    const output = await response.json();
    if (output) {
      uploadResponses.push(...output);
    }
  }
  return { files: uploadResponses };
}
const si = {
  radix: 1e3,
  unit: ["b", "kb", "Mb", "Gb", "Tb", "Pb", "Eb", "Zb", "Yb"]
};
const iec = {
  radix: 1024,
  unit: ["b", "Kib", "Mib", "Gib", "Tib", "Pib", "Eib", "Zib", "Yib"]
};
const jedec = {
  radix: 1024,
  unit: ["b", "Kb", "Mb", "Gb", "Tb", "Pb", "Eb", "Zb", "Yb"]
};
const SPECS = {
  si,
  iec,
  jedec
};
function filesize(bytes, fixed = 1, spec = "jedec") {
  bytes = Math.abs(bytes);
  const { radix, unit } = SPECS[spec] || SPECS.jedec;
  let loop = 0;
  while (bytes >= radix) {
    bytes /= radix;
    ++loop;
  }
  return `${bytes.toFixed(fixed)} ${unit[loop]}`;
}
async function upload(file_data, root_url, upload_id, max_file_size) {
  let files = (Array.isArray(file_data) ? file_data : [file_data]).map(
    (file_data2) => file_data2.blob
  );
  const oversized_files = files.filter(
    (f) => f.size > (max_file_size ?? Infinity)
  );
  if (oversized_files.length) {
    throw new Error(
      `File(s) exceed the maximum allowed size of ${filesize(max_file_size || Infinity)}: ${oversized_files.map((f) => `"${f.name}"`).join(", ")}`
    );
  }
  return await Promise.all(
    await this.upload_files(root_url, files, upload_id).then(
      async (response) => {
        if (response.error) {
          throw new Error(response.error);
        } else {
          if (response.files) {
            return response.files.map((f, i) => {
              const file2 = new FileData({
                ...file_data[i],
                path: f,
                url: `${root_url}${this.api_prefix}/file=${f}`
              });
              return file2;
            });
          }
          return [];
        }
      }
    )
  );
}
async function prepare_files(files, is_stream) {
  return files.map(
    (f) => new FileData({
      path: f.name,
      orig_name: f.name,
      blob: f,
      size: f.size,
      mime_type: f.type,
      is_stream
    })
  );
}
class FileData {
  path;
  url;
  orig_name;
  size;
  blob;
  is_stream;
  mime_type;
  alt_text;
  b64;
  meta = { _type: "gradio.FileData" };
  constructor({
    path,
    url,
    orig_name,
    size,
    blob,
    is_stream,
    mime_type,
    alt_text,
    b64
  }) {
    this.path = path;
    this.url = url;
    this.orig_name = orig_name;
    this.size = size;
    this.blob = url ? void 0 : blob;
    this.is_stream = is_stream;
    this.mime_type = mime_type;
    this.alt_text = alt_text;
    this.b64 = b64;
  }
}
class Command {
  type;
  command;
  meta;
  fileData;
  constructor(command, meta) {
    this.type = "command";
    this.command = command;
    this.meta = meta;
  }
}
typeof process !== "undefined" && process.versions && process.versions.node;
function update_object(object, newValue, stack) {
  while (stack.length > 1) {
    const key2 = stack.shift();
    if (typeof key2 === "string" || typeof key2 === "number") {
      object = object[key2];
    } else {
      throw new Error("Invalid key type");
    }
  }
  const key = stack.shift();
  if (typeof key === "string" || typeof key === "number") {
    object[key] = newValue;
  } else {
    throw new Error("Invalid key type");
  }
}
async function walk_and_store_blobs(data, type = void 0, path = [], root = false, endpoint_info = void 0) {
  if (Array.isArray(data)) {
    let blob_refs = [];
    await Promise.all(
      data.map(async (_, index) => {
        let new_path = path.slice();
        new_path.push(String(index));
        const array_refs = await walk_and_store_blobs(
          data[index],
          root ? endpoint_info?.parameters[index]?.component || void 0 : type,
          new_path,
          false,
          endpoint_info
        );
        blob_refs = blob_refs.concat(array_refs);
      })
    );
    return blob_refs;
  } else if (globalThis.Buffer && data instanceof globalThis.Buffer || data instanceof Blob) {
    return [
      {
        path,
        blob: new Blob([data]),
        type
      }
    ];
  } else if (typeof data === "object" && data !== null) {
    let blob_refs = [];
    for (const key of Object.keys(data)) {
      const new_path = [...path, key];
      const value = data[key];
      blob_refs = blob_refs.concat(
        await walk_and_store_blobs(
          value,
          void 0,
          new_path,
          false,
          endpoint_info
        )
      );
    }
    return blob_refs;
  }
  return [];
}
function skip_queue(id, config) {
  let fn_queue = config?.dependencies?.find((dep) => dep.id == id)?.queue;
  if (fn_queue != null) {
    return !fn_queue;
  }
  return !config.enable_queue;
}
function post_message(message, origin) {
  return new Promise((res, _rej) => {
    const channel = new MessageChannel();
    channel.port1.onmessage = (({ data }) => {
      channel.port1.close();
      res(data);
    });
    window.parent.postMessage(message, origin, [channel.port2]);
  });
}
function handle_payload(resolved_payload, dependency, components, type, with_null_state = false) {
  if (type === "input" && !with_null_state) {
    throw new Error("Invalid code path. Cannot skip state inputs for input.");
  }
  if (type === "output" && with_null_state) {
    return resolved_payload;
  }
  let updated_payload = [];
  let payload_index = 0;
  const deps = type === "input" ? dependency.inputs : dependency.outputs;
  for (let i = 0; i < deps.length; i++) {
    const input_id = deps[i];
    const component = components.find((c) => c.id === input_id);
    if (component?.type === "state") {
      if (with_null_state) {
        if (resolved_payload.length === deps.length) {
          const value = resolved_payload[payload_index];
          updated_payload.push(value);
          payload_index++;
        } else {
          updated_payload.push(null);
        }
      } else {
        payload_index++;
        continue;
      }
      continue;
    } else {
      const value = resolved_payload[payload_index];
      updated_payload.push(value);
      payload_index++;
    }
  }
  return updated_payload;
}
async function handle_blob(endpoint, data, api_info) {
  const self2 = this;
  await process_local_file_commands(self2, data);
  const blobRefs = await walk_and_store_blobs(
    data,
    void 0,
    [],
    true,
    api_info
  );
  const results = await Promise.all(
    blobRefs.map(async ({ path, blob, type }) => {
      if (!blob) return { path, type };
      const response = await self2.upload_files(endpoint, [blob]);
      const file_url = response.files && response.files[0];
      return {
        path,
        file_url,
        type,
        name: typeof File !== "undefined" && blob instanceof File ? blob?.name : void 0
      };
    })
  );
  results.forEach(({ path, file_url, type, name }) => {
    if (type === "Gallery") {
      update_object(data, file_url, path);
    } else if (file_url) {
      const file2 = new FileData({ path: file_url, orig_name: name });
      update_object(data, file2, path);
    }
  });
  return data;
}
async function process_local_file_commands(client, data) {
  const root = client.config?.root || client.config?.root_url;
  if (!root) {
    throw new Error(ROOT_URL_ERROR_MSG);
  }
  await recursively_process_commands(client, data);
}
async function recursively_process_commands(client, data, path = []) {
  for (const key in data) {
    if (data[key] instanceof Command) {
      await process_single_command(client, data, key);
    } else if (typeof data[key] === "object" && data[key] !== null) {
      await recursively_process_commands(client, data[key], [...path, key]);
    }
  }
}
async function process_single_command(client, data, key) {
  let cmd_item = data[key];
  const root = client.config?.root || client.config?.root_url;
  if (!root) {
    throw new Error(ROOT_URL_ERROR_MSG);
  }
  try {
    let fileBuffer;
    let fullPath;
    if (typeof process !== "undefined" && process.versions && process.versions.node) {
      const fs = await import('fs/promises');
      const path = await import('path');
      fullPath = path.resolve(process.cwd(), cmd_item.meta.path);
      fileBuffer = await fs.readFile(fullPath);
    } else {
      throw new Error(NODEJS_FS_ERROR_MSG);
    }
    const file2 = new Blob([fileBuffer], {
      type: "application/octet-stream"
    });
    const response = await client.upload_files(root, [file2]);
    const file_url = response.files && response.files[0];
    if (file_url) {
      const fileData = new FileData({
        path: file_url,
        orig_name: cmd_item.meta.name || ""
      });
      data[key] = fileData;
    }
  } catch (error) {
    console.error(FILE_PROCESSING_ERROR_MSG, error);
  }
}
async function post_data(url, body, additional_headers) {
  const headers = { "Content-Type": "application/json" };
  if (this.options.token) {
    headers.Authorization = `Bearer ${this.options.token}`;
  }
  try {
    var response = await this.fetch(url, {
      method: "POST",
      body: JSON.stringify(body),
      headers: { ...headers, ...additional_headers },
      credentials: "include"
    });
  } catch (e) {
    return [{ error: BROKEN_CONNECTION_MSG }, 500];
  }
  let output;
  let status;
  try {
    output = await response.json();
    status = response.status;
  } catch (e) {
    output = { error: `Could not parse server response: ${e}` };
    status = 500;
  }
  return [output, status];
}
async function predict(endpoint, data = {}) {
  let data_returned = false;
  let status_complete = false;
  if (!this.config) {
    throw new Error("Could not resolve app config");
  }
  if (typeof endpoint === "number") {
    this.config.dependencies.find((dep) => dep.id == endpoint);
  } else {
    const trimmed_endpoint = endpoint.replace(/^\//, "");
    this.config.dependencies.find(
      (dep) => dep.id == this.api_map[trimmed_endpoint]
    );
  }
  return new Promise(async (resolve, reject) => {
    const app = this.submit(endpoint, data, null, null, true);
    let result;
    for await (const message of app) {
      if (message.type === "data") {
        if (status_complete) {
          resolve(result);
        }
        data_returned = true;
        result = message;
      }
      if (message.type === "status") {
        if (message.stage === "error") reject(message);
        if (message.stage === "complete") {
          status_complete = true;
          if (data_returned) {
            resolve(result);
          }
        }
      }
    }
  });
}
async function check_space_status(id, type, status_callback) {
  let endpoint = type === "subdomain" ? `https://huggingface.co/api/spaces/by-subdomain/${id}` : `https://huggingface.co/api/spaces/${id}`;
  let response;
  let _status;
  try {
    response = await fetch(endpoint);
    _status = response.status;
    if (_status !== 200) {
      throw new Error();
    }
    response = await response.json();
  } catch (e) {
    status_callback({
      status: "error",
      load_status: "error",
      message: SPACE_STATUS_ERROR_MSG,
      detail: "NOT_FOUND"
    });
    return;
  }
  if (!response || _status !== 200) return;
  const {
    runtime: { stage },
    id: space_name
  } = response;
  switch (stage) {
    case "STOPPED":
    case "SLEEPING":
      status_callback({
        status: "sleeping",
        load_status: "pending",
        message: "Space is asleep. Waking it up...",
        detail: stage
      });
      setTimeout(() => {
        check_space_status(id, type, status_callback);
      }, 1e3);
      break;
    case "PAUSED":
      status_callback({
        status: "paused",
        load_status: "error",
        message: "This space has been paused by the author. If you would like to try this demo, consider duplicating the space.",
        detail: stage,
        discussions_enabled: await discussions_enabled(space_name)
      });
      break;
    case "RUNNING":
    case "RUNNING_BUILDING":
      status_callback({
        status: "running",
        load_status: "complete",
        message: "Space is running.",
        detail: stage
      });
      break;
    case "BUILDING":
      status_callback({
        status: "building",
        load_status: "pending",
        message: "Space is building...",
        detail: stage
      });
      setTimeout(() => {
        check_space_status(id, type, status_callback);
      }, 1e3);
      break;
    case "APP_STARTING":
      status_callback({
        status: "starting",
        load_status: "pending",
        message: "Space is starting...",
        detail: stage
      });
      setTimeout(() => {
        check_space_status(id, type, status_callback);
      }, 1e3);
      break;
    default:
      status_callback({
        status: "space_error",
        load_status: "error",
        message: "This space is experiencing an issue.",
        detail: stage,
        discussions_enabled: await discussions_enabled(space_name)
      });
      break;
  }
}
const check_and_wake_space = async (space_id, status_callback) => {
  let retries = 0;
  const max_retries = 12;
  const check_interval = 5e3;
  return new Promise((resolve) => {
    check_space_status(
      space_id,
      RE_SPACE_NAME.test(space_id) ? "space_name" : "subdomain",
      (status) => {
        status_callback(status);
        if (status.status === "running") {
          resolve();
        } else if (status.status === "error" || status.status === "paused" || status.status === "space_error") {
          resolve();
        } else if (status.status === "sleeping" || status.status === "building") {
          if (retries < max_retries) {
            retries++;
            setTimeout(() => {
              check_and_wake_space(space_id, status_callback).then(resolve);
            }, check_interval);
          } else {
            resolve();
          }
        }
      }
    );
  });
};
const RE_DISABLED_DISCUSSION = /^(?=[^]*\b[dD]iscussions{0,1}\b)(?=[^]*\b[dD]isabled\b)[^]*$/;
async function discussions_enabled(space_id) {
  try {
    const r = await fetch(
      `https://huggingface.co/api/spaces/${space_id}/discussions`,
      {
        method: "HEAD"
      }
    );
    const error = r.headers.get("x-error-message");
    if (!r.ok || error && RE_DISABLED_DISCUSSION.test(error)) return false;
    return true;
  } catch (e) {
    return false;
  }
}
async function get_space_hardware(space_id, token) {
  const headers = {};
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  try {
    const res = await fetch(
      `https://huggingface.co/api/spaces/${space_id}/${RUNTIME_URL}`,
      { headers }
    );
    if (res.status !== 200)
      throw new Error("Space hardware could not be obtained.");
    const { hardware } = await res.json();
    return hardware.current;
  } catch (e) {
    throw new Error(e.message);
  }
}
async function set_space_timeout(space_id, timeout, token) {
  const headers = {};
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  const body = {
    seconds: timeout
  };
  try {
    const res = await fetch(
      `https://huggingface.co/api/spaces/${space_id}/${SLEEPTIME_URL}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", ...headers },
        body: JSON.stringify(body)
      }
    );
    if (res.status !== 200) {
      throw new Error(
        "Could not set sleep timeout on duplicated Space. Please visit *ADD HF LINK TO SETTINGS* to set a timeout manually to reduce billing charges."
      );
    }
    const response = await res.json();
    return response;
  } catch (e) {
    throw new Error(e.message);
  }
}
const hardware_types = [
  "cpu-basic",
  "cpu-upgrade",
  "cpu-xl",
  "t4-small",
  "t4-medium",
  "a10g-small",
  "a10g-large",
  "a10g-largex2",
  "a10g-largex4",
  "a100-large",
  "zero-a10g",
  "h100",
  "h100x8"
];
async function duplicate(app_reference, options2) {
  const { token, private: _private, hardware, timeout, auth } = options2;
  if (hardware && !hardware_types.includes(hardware)) {
    throw new Error(
      `Invalid hardware type provided. Valid types are: ${hardware_types.map((v) => `"${v}"`).join(",")}.`
    );
  }
  const { http_protocol, host } = await process_endpoint(app_reference, token);
  let cookies = null;
  if (auth) {
    const cookie_header = await get_cookie_header(
      http_protocol,
      host,
      auth,
      fetch
    );
    if (cookie_header) cookies = parse_and_set_cookies(cookie_header);
  }
  const headers = {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
    ...cookies ? { Cookie: cookies.join("; ") } : {}
  };
  const user = (await (await fetch(`https://huggingface.co/api/whoami-v2`, {
    headers
  })).json()).name;
  const space_name = app_reference.split("/")[1];
  const body = {
    repository: `${user}/${space_name}`
  };
  if (_private) {
    body.private = true;
  }
  let original_hardware;
  try {
    if (!hardware) {
      original_hardware = await get_space_hardware(app_reference, token);
    }
  } catch (e) {
    throw Error(SPACE_METADATA_ERROR_MSG + e.message);
  }
  const requested_hardware = hardware || original_hardware || "cpu-basic";
  body.hardware = requested_hardware;
  try {
    const response = await fetch(
      `https://huggingface.co/api/spaces/${app_reference}/duplicate`,
      {
        method: "POST",
        headers,
        body: JSON.stringify(body)
      }
    );
    if (response.status === 409) {
      try {
        const client = await Client.connect(`${user}/${space_name}`, options2);
        return client;
      } catch (error) {
        console.error("Failed to connect Client instance:", error);
        throw error;
      }
    } else if (response.status !== 200) {
      throw new Error(response.statusText);
    }
    const duplicated_space = await response.json();
    await set_space_timeout(`${user}/${space_name}`, timeout || 300, token);
    return await Client.connect(
      get_space_reference(duplicated_space.url),
      options2
    );
  } catch (e) {
    throw new Error(e);
  }
}
function get_space_reference(url) {
  const regex = /https:\/\/huggingface.co\/spaces\/([^/]+\/[^/]+)/;
  const match = url.match(regex);
  if (match) {
    return match[1];
  }
}
class TextLineStream extends TransformStream {
  #currentLine = "";
  /** Constructs a new instance. */
  constructor(options2 = { allowCR: false }) {
    super({
      transform: (chars, controller) => {
        chars = this.#currentLine + chars;
        while (true) {
          const lfIndex = chars.indexOf("\n");
          const crIndex = options2.allowCR ? chars.indexOf("\r") : -1;
          if (crIndex !== -1 && crIndex !== chars.length - 1 && (lfIndex === -1 || lfIndex - 1 > crIndex)) {
            controller.enqueue(chars.slice(0, crIndex));
            chars = chars.slice(crIndex + 1);
            continue;
          }
          if (lfIndex === -1)
            break;
          const endIndex = chars[lfIndex - 1] === "\r" ? lfIndex - 1 : lfIndex;
          controller.enqueue(chars.slice(0, endIndex));
          chars = chars.slice(lfIndex + 1);
        }
        this.#currentLine = chars;
      },
      flush: (controller) => {
        if (this.#currentLine === "")
          return;
        const currentLine = options2.allowCR && this.#currentLine.endsWith("\r") ? this.#currentLine.slice(0, -1) : this.#currentLine;
        controller.enqueue(currentLine);
      }
    });
  }
}
function stream$1(input) {
  let decoder = new TextDecoderStream();
  let split2 = new TextLineStream({ allowCR: true });
  return input.pipeThrough(decoder).pipeThrough(split2);
}
function split(input) {
  let rgx = /[:]\s*/;
  let match = rgx.exec(input);
  let idx = match && match.index;
  if (idx) {
    return [
      input.substring(0, idx),
      input.substring(idx + match[0].length)
    ];
  }
}
function fallback(headers, key, value) {
  let tmp = headers.get(key);
  if (!tmp)
    headers.set(key, value);
}
async function* events(res, signal) {
  if (!res.body)
    return;
  let iter = stream$1(res.body);
  let line, reader = iter.getReader();
  let event;
  for (; ; ) {
    if (signal && signal.aborted) {
      return reader.cancel();
    }
    line = await reader.read();
    if (line.done)
      return;
    if (!line.value) {
      if (event)
        yield event;
      event = void 0;
      continue;
    }
    let [field, value] = split(line.value) || [];
    if (field === "data") {
      event ||= {};
      event[field] = event[field] ? event[field] + "\n" + value : value;
    } else if (field === "event") {
      event ||= {};
      event[field] = value;
    } else if (field === "id") {
      event ||= {};
      event[field] = String(+value) === value ? +value : value;
    } else if (field === "retry") {
      event ||= {};
      event[field] = +value || void 0;
    }
  }
}
async function stream(input, init2) {
  let req = new Request(input, init2);
  fallback(req.headers, "Accept", "text/event-stream");
  fallback(req.headers, "Content-Type", "application/json");
  let r = await fetch(req);
  if (!r.ok)
    throw r;
  return events(r, req.signal);
}
async function open_stream() {
  let {
    event_callbacks,
    unclosed_events,
    pending_stream_messages,
    stream_status,
    config,
    jwt
  } = this;
  const that = this;
  if (!config) {
    throw new Error("Could not resolve app config");
  }
  stream_status.open = true;
  let stream2 = null;
  let params = new URLSearchParams({
    session_hash: this.session_hash
  }).toString();
  let url = new URL(`${config.root}${this.api_prefix}/${SSE_URL}?${params}`);
  if (jwt) {
    url.searchParams.set("__sign", jwt);
  }
  stream2 = this.stream(url);
  if (!stream2) {
    console.warn("Cannot connect to SSE endpoint: " + url.toString());
    return;
  }
  stream2.onmessage = async function(event) {
    let _data = JSON.parse(event.data);
    if (_data.msg === "close_stream") {
      close_stream(stream_status, that.abort_controller);
      return;
    }
    const event_id = _data.event_id;
    if (!event_id) {
      await Promise.all(
        Object.keys(event_callbacks).map(
          (event_id2) => event_callbacks[event_id2](_data)
        )
      );
    } else if (event_callbacks[event_id] && config) {
      if (_data.msg === "process_completed" && ["sse", "sse_v1", "sse_v2", "sse_v2.1", "sse_v3"].includes(
        config.protocol
      )) {
        unclosed_events.delete(event_id);
      }
      let fn = event_callbacks[event_id];
      if (typeof window !== "undefined" && typeof document !== "undefined") {
        setTimeout(fn, 0, _data);
      } else {
        fn(_data);
      }
    } else {
      if (!pending_stream_messages[event_id]) {
        pending_stream_messages[event_id] = [];
      }
      pending_stream_messages[event_id].push(_data);
    }
  };
  stream2.onerror = async function(e) {
    console.error(e);
    await Promise.all(
      Object.keys(event_callbacks).map(
        (event_id) => event_callbacks[event_id]({
          msg: "broken_connection",
          message: BROKEN_CONNECTION_MSG
        })
      )
    );
  };
}
function close_stream(stream_status, abort_controller) {
  if (stream_status) {
    stream_status.open = false;
    abort_controller?.abort();
  }
}
function apply_diff_stream(pending_diff_streams, event_id, data) {
  let is_first_generation = !pending_diff_streams[event_id];
  if (is_first_generation) {
    pending_diff_streams[event_id] = [];
    data.data.forEach((value, i) => {
      pending_diff_streams[event_id][i] = value;
    });
  } else {
    data.data.forEach((value, i) => {
      let new_data = apply_diff(pending_diff_streams[event_id][i], value);
      pending_diff_streams[event_id][i] = new_data;
      data.data[i] = new_data;
    });
  }
}
function apply_diff(obj, diff) {
  diff.forEach(([action, path, value]) => {
    obj = apply_edit(obj, path, action, value);
  });
  return obj;
}
function apply_edit(target, path, action, value) {
  if (path.length === 0) {
    if (action === "replace") {
      return value;
    } else if (action === "append") {
      return target + value;
    }
    throw new Error(`Unsupported action: ${action}`);
  }
  let current2 = target;
  for (let i = 0; i < path.length - 1; i++) {
    current2 = current2[path[i]];
  }
  const last_path = path[path.length - 1];
  switch (action) {
    case "replace":
      current2[last_path] = value;
      break;
    case "append":
      current2[last_path] += value;
      break;
    case "add":
      if (Array.isArray(current2)) {
        current2.splice(Number(last_path), 0, value);
      } else {
        current2[last_path] = value;
      }
      break;
    case "delete":
      if (Array.isArray(current2)) {
        current2.splice(Number(last_path), 1);
      } else {
        delete current2[last_path];
      }
      break;
    default:
      throw new Error(`Unknown action: ${action}`);
  }
  return target;
}
function readable_stream(input, init2 = {}) {
  const instance = {
    close: () => {
      console.warn("Method not implemented.");
    },
    onerror: null,
    onmessage: null,
    onopen: null,
    readyState: 0,
    url: input.toString(),
    withCredentials: false,
    CONNECTING: 0,
    OPEN: 1,
    CLOSED: 2,
    addEventListener: () => {
      throw new Error("Method not implemented.");
    },
    dispatchEvent: () => {
      throw new Error("Method not implemented.");
    },
    removeEventListener: () => {
      throw new Error("Method not implemented.");
    }
  };
  stream(input, init2).then(async (res) => {
    instance.readyState = instance.OPEN;
    try {
      for await (const chunk of res) {
        instance.onmessage && instance.onmessage(chunk);
      }
      instance.readyState = instance.CLOSED;
    } catch (e) {
      instance.onerror && instance.onerror(e);
      instance.readyState = instance.CLOSED;
    }
  }).catch((e) => {
    console.error(e);
    instance.onerror && instance.onerror(e);
    instance.readyState = instance.CLOSED;
  });
  return instance;
}
function submit(endpoint, data = {}, event_data, trigger_id, all_events, additional_headers) {
  try {
    let fire_event = function(event) {
      if (all_events || events_to_publish[event.type]) {
        push_event(event);
      }
    }, close = function() {
      done = true;
      while (resolvers.length > 0)
        resolvers.shift()({
          value: void 0,
          done: true
        });
    }, push = function(data2) {
      if (resolvers.length > 0) {
        resolvers.shift()(data2);
      } else {
        values.push(data2);
      }
    }, push_error = function(error) {
      push(thenable_reject(error));
      close();
    }, push_event = function(event) {
      push({ value: event, done: false });
    }, next = function() {
      if (values.length > 0) {
        return Promise.resolve(values.shift());
      }
      return new Promise((resolve) => resolvers.push(resolve));
    };
    const { token } = this.options;
    const {
      fetch: fetch2,
      app_reference,
      config,
      session_hash,
      api_info,
      api_map,
      stream_status,
      pending_stream_messages,
      pending_diff_streams,
      event_callbacks,
      unclosed_events,
      post_data: post_data2,
      options: options2,
      api_prefix
    } = this;
    const addt_headers = additional_headers || { "x-gradio-user": "api" };
    const that = this;
    if (!api_info) throw new Error("No API found");
    if (!config) throw new Error("Could not resolve app config");
    let { fn_index, endpoint_info, dependency } = get_endpoint_info(
      api_info,
      endpoint,
      api_map,
      config
    );
    let resolved_data = map_data_to_params(data, endpoint_info);
    let stream2;
    let protocol = config.protocol ?? "ws";
    if (protocol === "ws") {
      throw new Error("WebSocket protocol is not supported in this version");
    }
    let event_id_final = "";
    let event_id_cb = () => event_id_final;
    const _endpoint = typeof endpoint === "number" ? "/predict" : endpoint;
    let payload;
    let event_id = null;
    let complete = false;
    let last_status = {};
    let url_params = typeof window !== "undefined" && typeof document !== "undefined" ? new URLSearchParams(window.location.search).toString() : "";
    const events_to_publish = options2?.events?.reduce(
      (acc, event) => {
        acc[event] = true;
        return acc;
      },
      {}
    ) || {};
    async function cancel() {
      let reset_request = {};
      let cancel_request = {};
      reset_request = { event_id };
      cancel_request = { event_id, session_hash, fn_index };
      try {
        if (!config) {
          throw new Error("Could not resolve app config");
        }
        if ("event_id" in cancel_request) {
          await fetch2(`${config.root}${api_prefix}/${CANCEL_URL}`, {
            headers: { "Content-Type": "application/json" },
            method: "POST",
            body: JSON.stringify(cancel_request)
          });
        }
        await fetch2(`${config.root}${api_prefix}/${RESET_URL}`, {
          headers: { "Content-Type": "application/json" },
          method: "POST",
          body: JSON.stringify(reset_request)
        });
      } catch (e) {
        console.warn(
          "The `/reset` endpoint could not be called. Subsequent endpoint results may be unreliable."
        );
      }
    }
    const resolve_heartbeat = async (config2) => {
      await this._resolve_heartbeat(config2);
    };
    async function handle_render_config(render_config) {
      if (!config) return;
      let render_id = render_config.render_id;
      config.components = [
        ...config.components.filter((c) => c.props.rendered_in !== render_id),
        ...render_config.components
      ];
      config.dependencies = [
        ...config.dependencies.filter((d) => d.rendered_in !== render_id),
        ...render_config.dependencies
      ];
      const any_state = config.components.some((c) => c.type === "state");
      const any_unload = config.dependencies.some(
        (d) => d.targets.some((t) => t[1] === "unload")
      );
      config.connect_heartbeat = any_state || any_unload;
      await resolve_heartbeat(config);
      fire_event({
        type: "render",
        data: render_config,
        endpoint: _endpoint,
        fn_index
      });
    }
    const job = this.handle_blob(
      config.root,
      resolved_data,
      endpoint_info
    ).then(async (_payload) => {
      let input_data = handle_payload(
        _payload,
        dependency,
        config.components,
        "input",
        true
      );
      payload = {
        data: input_data || [],
        event_data,
        fn_index,
        trigger_id
      };
      if (skip_queue(fn_index, config)) {
        fire_event({
          type: "status",
          endpoint: _endpoint,
          stage: "pending",
          queue: false,
          fn_index,
          time: /* @__PURE__ */ new Date()
        });
        post_data2(
          `${config.root}${api_prefix}/run${_endpoint.startsWith("/") ? _endpoint : `/${_endpoint}`}${url_params ? "?" + url_params : ""}`,
          {
            ...payload,
            session_hash
          },
          addt_headers
        ).then(async ([output, status_code]) => {
          const data2 = output.data;
          if (status_code == 200) {
            fire_event({
              type: "data",
              endpoint: _endpoint,
              fn_index,
              data: handle_payload(
                data2,
                dependency,
                config.components,
                "output",
                options2.with_null_state
              ),
              time: /* @__PURE__ */ new Date(),
              event_data,
              trigger_id
            });
            if (output.render_config) {
              await handle_render_config(output.render_config);
            }
            fire_event({
              type: "status",
              endpoint: _endpoint,
              fn_index,
              stage: "complete",
              eta: output.average_duration,
              queue: false,
              time: /* @__PURE__ */ new Date()
            });
          } else {
            const is_connection_error = output?.error === BROKEN_CONNECTION_MSG;
            fire_event({
              type: "status",
              stage: "error",
              endpoint: _endpoint,
              fn_index,
              message: output.error,
              broken: is_connection_error,
              queue: false,
              time: /* @__PURE__ */ new Date()
            });
          }
        }).catch((e) => {
          fire_event({
            type: "status",
            stage: "error",
            message: e.message,
            endpoint: _endpoint,
            fn_index,
            queue: false,
            time: /* @__PURE__ */ new Date()
          });
        });
      } else if (protocol == "sse") {
        fire_event({
          type: "status",
          stage: "pending",
          queue: true,
          endpoint: _endpoint,
          fn_index,
          time: /* @__PURE__ */ new Date()
        });
        var params = new URLSearchParams({
          fn_index: fn_index.toString(),
          session_hash
        }).toString();
        let url = new URL(
          `${config.root}${api_prefix}/${SSE_URL}?${url_params ? url_params + "&" : ""}${params}`
        );
        if (this.jwt) {
          url.searchParams.set("__sign", this.jwt);
        }
        stream2 = this.stream(url);
        if (!stream2) {
          return Promise.reject(
            new Error("Cannot connect to SSE endpoint: " + url.toString())
          );
        }
        stream2.onmessage = async function(event) {
          const _data = JSON.parse(event.data);
          const { type, status, data: data2 } = handle_message(
            _data,
            last_status[fn_index]
          );
          if (type === "update" && status && !complete) {
            fire_event({
              type: "status",
              endpoint: _endpoint,
              fn_index,
              time: /* @__PURE__ */ new Date(),
              ...status
            });
            if (status.stage === "error") {
              stream2?.close();
              close();
            }
          } else if (type === "data") {
            let [_, status2] = await post_data2(
              `${config.root}${api_prefix}/queue/data`,
              {
                ...payload,
                session_hash,
                event_id
              }
            );
            if (status2 !== 200) {
              fire_event({
                type: "status",
                stage: "error",
                message: BROKEN_CONNECTION_MSG,
                queue: true,
                endpoint: _endpoint,
                fn_index,
                time: /* @__PURE__ */ new Date()
              });
              stream2?.close();
              close();
            }
          } else if (type === "complete") {
            complete = status;
          } else if (type === "log") {
            fire_event({
              type: "log",
              title: data2.title,
              log: data2.log,
              level: data2.level,
              endpoint: _endpoint,
              duration: data2.duration,
              visible: data2.visible,
              fn_index
            });
          } else if (type === "generating" || type === "streaming") {
            fire_event({
              type: "status",
              time: /* @__PURE__ */ new Date(),
              ...status,
              stage: status?.stage,
              queue: true,
              endpoint: _endpoint,
              fn_index
            });
          }
          if (data2) {
            fire_event({
              type: "data",
              time: /* @__PURE__ */ new Date(),
              data: handle_payload(
                data2.data,
                dependency,
                config.components,
                "output",
                options2.with_null_state
              ),
              endpoint: _endpoint,
              fn_index,
              event_data,
              trigger_id
            });
            if (complete) {
              fire_event({
                type: "status",
                time: /* @__PURE__ */ new Date(),
                ...complete,
                stage: status?.stage,
                queue: true,
                endpoint: _endpoint,
                fn_index
              });
              stream2?.close();
              close();
            }
          }
        };
      } else if (protocol == "sse_v1" || protocol == "sse_v2" || protocol == "sse_v2.1" || protocol == "sse_v3") {
        fire_event({
          type: "status",
          stage: "pending",
          queue: true,
          endpoint: _endpoint,
          fn_index,
          time: /* @__PURE__ */ new Date()
        });
        let hostname = "";
        if (typeof window !== "undefined" && typeof document !== "undefined") {
          hostname = window?.location?.hostname;
        }
        let hfhubdev = "dev.spaces.huggingface.tech";
        const origin = hostname.includes(".dev.") ? `https://moon-${hostname.split(".")[1]}.${hfhubdev}` : `https://huggingface.co`;
        const is_zerogpu_iframe = typeof window !== "undefined" && typeof document !== "undefined" && window.parent != window && window.supports_zerogpu_headers;
        const zerogpu_auth_promise = is_zerogpu_iframe ? post_message("zerogpu-headers", origin) : Promise.resolve(null);
        const post_data_promise = zerogpu_auth_promise.then((headers) => {
          const combined_headers = { ...addt_headers, ...headers || {} };
          return post_data2(
            `${config.root}${api_prefix}/${SSE_DATA_URL}?${url_params}`,
            {
              ...payload,
              session_hash
            },
            combined_headers
          );
        });
        return post_data_promise.then(async ([response, status]) => {
          if (response.event_id) {
            event_id_final = response.event_id;
          }
          if (status === 503) {
            fire_event({
              type: "status",
              stage: "error",
              message: QUEUE_FULL_MSG,
              queue: true,
              endpoint: _endpoint,
              fn_index,
              time: /* @__PURE__ */ new Date(),
              visible: true
            });
          } else if (status === 422) {
            fire_event({
              type: "status",
              stage: "error",
              message: response.detail,
              queue: true,
              endpoint: _endpoint,
              fn_index,
              code: "validation_error",
              time: /* @__PURE__ */ new Date(),
              visible: true
            });
            close();
          } else if (status !== 200) {
            const is_connection_error = response?.error === BROKEN_CONNECTION_MSG;
            fire_event({
              type: "status",
              stage: "error",
              broken: is_connection_error,
              message: is_connection_error ? BROKEN_CONNECTION_MSG : response.detail || response.error,
              queue: true,
              endpoint: _endpoint,
              fn_index,
              time: /* @__PURE__ */ new Date(),
              visible: true
            });
          } else {
            event_id = response.event_id;
            event_id_final = event_id;
            let callback = async function(_data) {
              try {
                const { type, status: status2, data: data2, original_msg } = handle_message(
                  _data,
                  last_status[fn_index]
                );
                if (type == "heartbeat") {
                  return;
                }
                if (type === "update" && status2 && !complete) {
                  fire_event({
                    type: "status",
                    endpoint: _endpoint,
                    fn_index,
                    time: /* @__PURE__ */ new Date(),
                    original_msg,
                    ...status2
                  });
                } else if (type === "complete") {
                  complete = status2;
                } else if (type == "unexpected_error" || type == "broken_connection") {
                  console.error("Unexpected error", status2?.message);
                  const broken = type === "broken_connection";
                  fire_event({
                    type: "status",
                    stage: "error",
                    message: status2?.message || "An Unexpected Error Occurred!",
                    queue: true,
                    endpoint: _endpoint,
                    broken,
                    session_not_found: status2?.session_not_found,
                    fn_index,
                    time: /* @__PURE__ */ new Date()
                  });
                } else if (type === "log") {
                  fire_event({
                    type: "log",
                    title: data2.title,
                    log: data2.log,
                    level: data2.level,
                    endpoint: _endpoint,
                    duration: data2.duration,
                    visible: data2.visible,
                    fn_index
                  });
                  return;
                } else if (type === "generating" || type === "streaming") {
                  fire_event({
                    type: "status",
                    time: /* @__PURE__ */ new Date(),
                    ...status2,
                    stage: status2?.stage,
                    queue: true,
                    endpoint: _endpoint,
                    fn_index
                  });
                  if (data2 && dependency.connection !== "stream" && ["sse_v2", "sse_v2.1", "sse_v3"].includes(protocol)) {
                    apply_diff_stream(pending_diff_streams, event_id, data2);
                  }
                }
                if (data2) {
                  fire_event({
                    type: "data",
                    time: /* @__PURE__ */ new Date(),
                    data: handle_payload(
                      data2.data,
                      dependency,
                      config.components,
                      "output",
                      options2.with_null_state
                    ),
                    endpoint: _endpoint,
                    fn_index
                  });
                  if (data2.render_config) {
                    await handle_render_config(data2.render_config);
                  }
                  if (complete) {
                    fire_event({
                      type: "status",
                      time: /* @__PURE__ */ new Date(),
                      ...complete,
                      stage: status2?.stage,
                      queue: true,
                      endpoint: _endpoint,
                      fn_index
                    });
                    close();
                  }
                }
                if (status2?.stage === "complete" || status2?.stage === "error") {
                  if (event_callbacks[event_id]) {
                    delete event_callbacks[event_id];
                  }
                  if (event_id in pending_diff_streams) {
                    delete pending_diff_streams[event_id];
                  }
                }
              } catch (e) {
                console.error("Unexpected client exception", e);
                fire_event({
                  type: "status",
                  stage: "error",
                  message: "An Unexpected Error Occurred!",
                  queue: true,
                  endpoint: _endpoint,
                  fn_index,
                  time: /* @__PURE__ */ new Date()
                });
                if (["sse_v2", "sse_v2.1", "sse_v3"].includes(protocol)) {
                  close_stream(stream_status, that.abort_controller);
                  stream_status.open = false;
                  close();
                }
              }
            };
            if (event_id in pending_stream_messages) {
              pending_stream_messages[event_id].forEach((msg) => callback(msg));
              delete pending_stream_messages[event_id];
            }
            event_callbacks[event_id] = callback;
            unclosed_events.add(event_id);
            if (!stream_status.open) {
              await this.open_stream();
            }
          }
        });
      }
    });
    let done = false;
    const values = [];
    const resolvers = [];
    const iterator = {
      [Symbol.asyncIterator]: () => iterator,
      next,
      throw: async (value) => {
        push_error(value);
        return next();
      },
      return: async () => {
        close();
        return { value: void 0, done: true };
      },
      cancel,
      send_chunk: (payload2) => {
        this.post_data(`${config.root}${api_prefix}/stream/${event_id_final}`, {
          ...payload2,
          session_hash: this.session_hash
        });
      },
      close_stream: () => {
        this.post_data(
          `${config.root}${api_prefix}/stream/${event_id_final}/close`,
          {}
        );
        close();
      },
      event_id: () => event_id_final,
      wait_for_id: async () => {
        await job;
        return event_id;
      }
    };
    return iterator;
  } catch (error) {
    console.error("Submit function encountered an error:", error);
    throw error;
  }
}
function thenable_reject(error) {
  return {
    then: (resolve, reject) => reject(error)
  };
}
function get_endpoint_info(api_info, endpoint, api_map, config) {
  let fn_index;
  let endpoint_info;
  let dependency;
  if (typeof endpoint === "number") {
    fn_index = endpoint;
    endpoint_info = api_info.unnamed_endpoints[fn_index];
    dependency = config.dependencies.find((dep) => dep.id == endpoint);
  } else {
    const trimmed_endpoint = endpoint.replace(/^\//, "");
    fn_index = api_map[trimmed_endpoint];
    endpoint_info = api_info.named_endpoints[endpoint.trim()];
    dependency = config.dependencies.find(
      (dep) => dep.id == api_map[trimmed_endpoint]
    );
  }
  if (typeof fn_index !== "number") {
    throw new Error(
      "There is no endpoint matching that name of fn_index matching that number."
    );
  }
  return { fn_index, endpoint_info, dependency };
}
class Client {
  app_reference;
  options;
  deep_link = null;
  config;
  api_prefix = "";
  api_info;
  api_map = {};
  session_hash = Math.random().toString(36).substring(2);
  jwt = false;
  last_status = {};
  cookies = null;
  // streaming
  stream_status = { open: false };
  closed = false;
  pending_stream_messages = {};
  pending_diff_streams = {};
  event_callbacks = {};
  unclosed_events = /* @__PURE__ */ new Set();
  heartbeat_event = null;
  abort_controller = null;
  stream_instance = null;
  current_payload;
  get_url_config(url = null) {
    if (!this.config) {
      throw new Error(CONFIG_ERROR_MSG);
    }
    if (url === null) {
      url = window.location.href;
    }
    const stripSlashes = (str) => str.replace(/^\/+|\/+$/g, "");
    let root_path = stripSlashes(new URL(this.config.root).pathname);
    let url_path = stripSlashes(new URL(url).pathname);
    let page;
    if (!url_path.startsWith(root_path)) {
      page = "";
    } else {
      page = stripSlashes(url_path.substring(root_path.length));
    }
    return this.get_page_config(page);
  }
  get_page_config(page) {
    if (!this.config) {
      throw new Error(CONFIG_ERROR_MSG);
    }
    let config = this.config;
    if (!(page in config.page)) {
      page = "";
    }
    return {
      ...config,
      current_page: page,
      layout: config.page[page].layout,
      components: config.components.filter(
        (c) => config.page[page].components.includes(c.id)
      ),
      dependencies: this.config.dependencies.filter(
        (d) => config.page[page].dependencies.includes(d.id)
      )
    };
  }
  fetch(input, init2) {
    const headers = new Headers(init2?.headers || {});
    if (this && this.cookies) {
      headers.append("Cookie", this.cookies);
    }
    if (this && this.options.headers) {
      let additional_headers = new Headers(this.options.headers);
      additional_headers.forEach((value, name) => {
        headers.append(name, value);
      });
    }
    return fetch(input, { ...init2, headers });
  }
  stream(url) {
    const headers = new Headers();
    if (this && this.cookies) {
      headers.append("Cookie", this.cookies);
    }
    if (this && this.options.headers) {
      let additional_headers = new Headers(this.options.headers);
      additional_headers.forEach((value, name) => {
        headers.append(name, value);
      });
    }
    if (this && this.options.token) {
      headers.append("Authorization", `Bearer ${this.options.token}`);
    }
    this.abort_controller = new AbortController();
    this.stream_instance = readable_stream(url.toString(), {
      credentials: "include",
      headers,
      signal: this.abort_controller.signal
    });
    return this.stream_instance;
  }
  view_api;
  upload_files;
  upload;
  handle_blob;
  post_data;
  submit;
  predict;
  open_stream;
  resolve_config;
  resolve_cookies;
  constructor(app_reference, options2 = { events: ["data"] }) {
    this.app_reference = app_reference;
    this.deep_link = options2.query_params?.deep_link || null;
    if (!options2.events) {
      options2.events = ["data"];
    }
    this.options = options2;
    this.current_payload = {};
    if (options2.cookies) {
      this.cookies = options2.cookies;
    }
    this.view_api = view_api.bind(this);
    this.upload_files = upload_files.bind(this);
    this.handle_blob = handle_blob.bind(this);
    this.post_data = post_data.bind(this);
    this.submit = submit.bind(this);
    this.predict = predict.bind(this);
    this.open_stream = open_stream.bind(this);
    this.resolve_config = resolve_config.bind(this);
    this.resolve_cookies = resolve_cookies.bind(this);
    this.upload = upload.bind(this);
    this.fetch = this.fetch.bind(this);
    this.handle_space_success = this.handle_space_success.bind(this);
    this.stream = this.stream.bind(this);
  }
  async init() {
    if (this.options.auth) {
      await this.resolve_cookies();
    }
    await this._resolve_config().then(
      ({ config }) => this._resolve_heartbeat(config)
    );
    this.api_info = await this.view_api();
    this.api_map = map_names_to_ids(this.config?.dependencies || []);
  }
  async _resolve_heartbeat(_config) {
    if (_config) {
      this.config = _config;
      this.api_prefix = _config.api_prefix || "";
      if (this.config && this.config.connect_heartbeat) {
        if (this.config.space_id && this.options.token) {
          this.jwt = await get_jwt(
            this.config.space_id,
            this.options.token,
            this.cookies
          );
        }
      }
    }
    if (_config.space_id && this.options.token) {
      this.jwt = await get_jwt(_config.space_id, this.options.token);
    }
    if (this.config && this.config.connect_heartbeat) {
      const heartbeat_url = new URL(
        `${this.config.root}${this.api_prefix}/${HEARTBEAT_URL}/${this.session_hash}`
      );
      if (this.jwt) {
        heartbeat_url.searchParams.set("__sign", this.jwt);
      }
      if (!this.heartbeat_event) {
        this.heartbeat_event = this.stream(heartbeat_url);
      }
    }
  }
  static async connect(app_reference, options2 = {
    events: ["data"]
  }) {
    const client2 = new this(app_reference, options2);
    if (options2.session_hash) {
      client2.session_hash = options2.session_hash;
    }
    await client2.init();
    return client2;
  }
  async reconnect() {
    const app_id_url = new URL(
      `${this.config.root}${this.api_prefix}/${APP_ID_URL}`
    );
    let app_id;
    try {
      const response = await this.fetch(app_id_url);
      if (!response.ok) {
        throw new Error();
      }
      app_id = (await response.json()).app_id;
    } catch (e) {
      return "broken";
    }
    if (app_id !== this.config.app_id) {
      return "changed";
    }
    return "connected";
  }
  close() {
    this.closed = true;
    close_stream(this.stream_status, this.abort_controller);
  }
  set_current_payload(payload) {
    this.current_payload = payload;
  }
  static async duplicate(app_reference, options2 = {
    events: ["data"]
  }) {
    return duplicate(app_reference, options2);
  }
  async _resolve_config() {
    const { http_protocol, host, space_id } = await process_endpoint(
      this.app_reference,
      this.options.token
    );
    const { status_callback } = this.options;
    if (space_id && status_callback) {
      await check_and_wake_space(space_id, status_callback);
    }
    let config;
    try {
      let configUrl = `${http_protocol}//${host}`;
      config = await this.resolve_config(configUrl);
      if (!config) {
        throw new Error(CONFIG_ERROR_MSG);
      }
      return this.config_success(config);
    } catch (e) {
      if (space_id && status_callback) {
        check_space_status(
          space_id,
          RE_SPACE_NAME.test(space_id) ? "space_name" : "subdomain",
          this.handle_space_success
        );
      } else {
        if (status_callback)
          status_callback({
            status: "error",
            message: "Could not load this space.",
            load_status: "error",
            detail: "NOT_FOUND"
          });
        throw Error(e);
      }
    }
  }
  async config_success(_config) {
    this.config = _config;
    this.api_prefix = _config.api_prefix || "";
    if (this.config.auth_required) {
      return this.prepare_return_obj();
    }
    try {
      this.api_info = await this.view_api();
    } catch (e) {
      console.error(API_INFO_ERROR_MSG + e.message);
    }
    return this.prepare_return_obj();
  }
  async handle_space_success(status) {
    if (!this) {
      throw new Error(CONFIG_ERROR_MSG);
    }
    const { status_callback } = this.options;
    if (status_callback) status_callback(status);
    if (status.status === "running") {
      try {
        this.config = await this._resolve_config();
        this.api_prefix = this?.config?.api_prefix || "";
        if (!this.config) {
          throw new Error(CONFIG_ERROR_MSG);
        }
        const _config = await this.config_success(this.config);
        return _config;
      } catch (e) {
        if (status_callback) {
          status_callback({
            status: "error",
            message: "Could not load this space.",
            load_status: "error",
            detail: "NOT_FOUND"
          });
        }
        throw e;
      }
    }
  }
  async component_server(component_id, fn_name, data) {
    if (!this.config) {
      throw new Error(CONFIG_ERROR_MSG);
    }
    const headers = {};
    const { token } = this.options;
    const { session_hash } = this;
    if (token) {
      headers.Authorization = `Bearer ${this.options.token}`;
    }
    let root_url;
    let component = this.config.components.find(
      (comp) => comp.id === component_id
    );
    if (component?.props?.root_url) {
      root_url = component.props.root_url;
    } else {
      root_url = this.config.root;
    }
    let body;
    if (typeof data === "object" && data !== null && "binary" in data) {
      const _data = data;
      body = new FormData();
      for (const key in _data.data) {
        if (key === "binary") continue;
        body.append(key, _data.data[key]);
      }
      body.set("component_id", component_id.toString());
      body.set("fn_name", fn_name);
      body.set("session_hash", session_hash);
    } else {
      body = JSON.stringify({
        data,
        component_id,
        fn_name,
        session_hash
      });
      headers["Content-Type"] = "application/json";
    }
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
    try {
      const response = await this.fetch(
        `${root_url}${this.api_prefix}/${COMPONENT_SERVER_URL}/`,
        {
          method: "POST",
          body,
          headers,
          credentials: "include"
        }
      );
      if (!response.ok) {
        throw new Error(
          "Could not connect to component server: " + response.statusText
        );
      }
      const output = await response.json();
      return output;
    } catch (e) {
      console.warn(e);
    }
  }
  set_cookies(raw_cookies) {
    this.cookies = parse_and_set_cookies(raw_cookies).join("; ");
  }
  prepare_return_obj() {
    return {
      config: this.config,
      predict: this.predict,
      submit: this.submit,
      view_api: this.view_api,
      component_server: this.component_server
    };
  }
}
const loading = {
  ar: "جاري التحميل",
  ca: "S'està carregant",
  ckb: "بارکردن",
  de: "Laden",
  en: "Loading",
  es: "Cargando",
  eu: "Kargatzen",
  fa: "در حال بارگذاری",
  fi: "Ladataan",
  fr: "Chargement",
  he: "טוען",
  hi: "लोड हो रहा है",
  ja: "読み込み中",
  ko: "로딩 중",
  lt: "Kraunama",
  nb: "Laster",
  nl: "Laden",
  pl: "Ładowanie",
  "pt-BR": "Carregando",
  pt: "A carregar",
  ro: "Se încarcă",
  ru: "Загрузка",
  sv: "Laddar",
  ta: "ஏற்றுகிறது",
  th: "กำลังโหลด",
  tr: "Yükleniyor",
  uk: "Завантаження",
  ur: "لوڈ ہو رہا ہے",
  uz: "Yuklanmoqda",
  "zh-CN": "加载中",
  "zh-TW": "載入中"
};
const _name = "English";
const annotated_image = { "annotated_image": "Annotated Image" };
const audio = { "allow_recording_access": "Please allow access to the microphone for recording.", "audio": "Audio", "drop_to_upload": "Drop an audio file here to upload", "record_from_microphone": "Record from microphone", "stop_recording": "Stop recording", "no_device_support": "Media devices could not be accessed. Check that you are running on a secure origin (https) or localhost (or you have passed a valid SSL certificate to ssl_verify), and you have allowed browser access to your device.", "stop": "Stop", "resume": "Resume", "record": "Record", "no_microphone": "No microphone found", "pause": "Pause", "play": "Play", "waiting": "Waiting" };
const blocks = { "connection_can_break": "On mobile, the connection can break if this tab is unfocused or the device sleeps, losing your position in queue.", "long_requests_queue": "There is a long queue of requests pending. Duplicate this Space to skip.", "lost_connection": "Lost connection due to leaving page. Rejoining queue...", "waiting_for_inputs": "Waiting for file(s) to finish uploading, please retry." };
const chatbot = { "edit": "Edit", "retry": "Retry", "undo": "Undo", "submit": "Submit", "cancel": "Cancel", "like": "Like", "dislike": "Dislike", "clear": "Clear", "copy_message": "Copy message", "copied_message": "Copied message" };
const chat_interface = { "new_chat": "New chat", "message_placeholder": "Type a message...", "additional_inputs": "Additional inputs", "chatbot": "Chatbot", "conversation": "Conversation" };
const checkbox = { "checkbox": "Checkbox", "checkbox_group": "Checkbox Group" };
const code = { "code": "Code" };
const color_picker = { "color_picker": "Color Picker" };
const common = { "built_with": "built with", "built_with_gradio": "Built with Gradio", "clear": "Clear", "download": "Download", "edit": "Edit", "empty": "Empty", "error": "Error", "hosted_on": "Hosted on", "loading": "Loading", "logo": "logo", "or": "or", "remove": "Remove", "settings": "Settings", "share": "Share", "submit": "Submit", "undo": "Undo", "no_devices": "No devices found", "language": "Language", "display_theme": "Display Theme", "pwa": "Progressive Web App", "record": "Record", "stop_recording": "Stop Recording", "screen_studio": "Screen Studio", "share_gradio_tab": "[Sharing] Gradio Tab", "run": "Run" };
const dataframe = { "incorrect_format": "Incorrect format, only CSV and TSV files are supported", "new_column": "Add column", "new_row": "New row", "add_row_above": "Add row above", "add_row_below": "Add row below", "delete_row": "Delete row", "delete_column": "Delete column", "add_column_left": "Add column to the left", "add_column_right": "Add column to the right", "sort_column": "Sort column", "sort_ascending": "Sort ascending", "sort_descending": "Sort descending", "drop_to_upload": "Drop CSV or TSV files here to import data into dataframe", "clear_sort": "Clear sort", "filter": "Filter", "clear_filter": "Clear filters" };
const dropdown = { "dropdown": "Dropdown" };
const errors = { "build_error": "there is a build error", "config_error": "there is a config error", "contact_page_author": "Please contact the author of the page to let them know.", "no_app_file": "there is no app file", "runtime_error": "there is a runtime error", "space_not_working": `"Space isn't working because" {0}`, "space_paused": "the space is paused", "use_via_api": "Use via API", "use_via_api_or_mcp": "Use via API or MCP" };
const file = { "uploading": "Uploading..." };
const highlighted_text = { "highlighted_text": "Highlighted Text" };
const image = { "allow_webcam_access": "Please allow access to the webcam for recording.", "brush_color": "Brush color", "brush_radius": "Brush radius", "image": "Image", "remove_image": "Remove Image", "select_brush_color": "Select brush color", "start_drawing": "Start drawing", "use_brush": "Use brush", "drop_to_upload": "Drop an image file here to upload" };
const label = { "label": "Label" };
const login = { "enable_cookies": "If you are visiting a HuggingFace Space in Incognito mode, you must enable third party cookies.", "incorrect_credentials": "Incorrect Credentials", "username": "username", "password": "password", "login": "Login" };
const number = { "number": "Number" };
const plot = { "plot": "Plot" };
const radio = { "radio": "Radio" };
const slider = { "slider": "Slider" };
const upload_text = { "click_to_upload": "Click to Upload", "drop_audio": "Drop Audio Here", "drop_csv": "Drop CSV Here", "drop_file": "Drop File Here", "drop_image": "Drop Image Here", "drop_video": "Drop Video Here", "drop_gallery": "Drop Media Here", "paste_clipboard": "Paste from Clipboard" };
const video = { "drop_to_upload": "Drop a video file here to upload" };
const en = {
  _name,
  "3D_model": { "3d_model": "3D Model", "drop_to_upload": "Drop a 3D model (.obj, .glb, .stl, .gltf, .splat, or .ply) file here to upload" },
  annotated_image,
  audio,
  blocks,
  chatbot,
  chat_interface,
  checkbox,
  code,
  color_picker,
  common,
  dataframe,
  dropdown,
  errors,
  file,
  highlighted_text,
  image,
  label,
  login,
  number,
  plot,
  radio,
  slider,
  upload_text,
  video
};
const en$1 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  _name,
  annotated_image,
  audio,
  blocks,
  chat_interface,
  chatbot,
  checkbox,
  code,
  color_picker,
  common,
  dataframe,
  default: en,
  dropdown,
  errors,
  file,
  highlighted_text,
  image,
  label,
  login,
  number,
  plot,
  radio,
  slider,
  upload_text,
  video
}, Symbol.toStringTag, { value: "Module" }));
const lang_map = {
  ar: "العربية",
  ca: "Català",
  ckb: "کوردی",
  de: "Deutsch",
  en: "English",
  es: "Español",
  eu: "Euskara",
  fa: "فارسی",
  fi: "Suomi",
  fr: "Français",
  he: "עברית",
  hi: "हिंदी",
  ja: "日本語",
  ko: "한국어",
  lt: "Lietuvių",
  nb: "Norsk bokmål",
  nl: "Nederlands",
  pl: "Polski",
  "pt-BR": "Português do Brasil",
  pt: "Português",
  ro: "Română",
  ru: "Русский",
  sv: "Svenska",
  ta: "தமிழ்",
  th: "ภาษาไทย",
  tr: "Türkçe",
  uk: "Українська",
  ur: "اردو",
  uz: "O'zbek",
  "zh-CN": "简体中文",
  "zh-TW": "繁體中文"
};
const langs = /* @__PURE__ */ Object.assign({ "./lang/ar.json": () => import('./ar-DUjhs9at.js'), "./lang/ca.json": () => import('./ca-xGgXl-3Z.js'), "./lang/ckb.json": () => import('./ckb-ChERRNuj.js'), "./lang/de.json": () => import('./de-FmwryDR7.js'), "./lang/en.json": () => Promise.resolve().then(() => en$1), "./lang/es.json": () => import('./es-BE9jijsg.js'), "./lang/eu.json": () => import('./eu-wxr0YHHC.js'), "./lang/fa.json": () => import('./fa-BjiQAF52.js'), "./lang/fi.json": () => import('./fi-BGbOKfoP.js'), "./lang/fr.json": () => import('./fr-2gCwu8h7.js'), "./lang/he.json": () => import('./he-DklLcpNM.js'), "./lang/hi.json": () => import('./hi-C4p28L1P.js'), "./lang/id.json": () => import('./id-DmBF7ABE.js'), "./lang/ja.json": () => import('./ja-Cv5E0QTk.js'), "./lang/ko.json": () => import('./ko-Dk9HZuJp.js'), "./lang/lt.json": () => import('./lt-DWzhxrRN.js'), "./lang/nb.json": () => import('./nb-DEqxD3Ip.js'), "./lang/nl.json": () => import('./nl-CVFxJ9TU.js'), "./lang/pl.json": () => import('./pl-DgIjXIEn.js'), "./lang/pt-BR.json": () => import('./pt-BR-Cwe5l-9s.js'), "./lang/pt.json": () => import('./pt-DQPoVyIw.js'), "./lang/ro.json": () => import('./ro-RF9EIh6L.js'), "./lang/ru.json": () => import('./ru-Cuydnn8h.js'), "./lang/sv.json": () => import('./sv-DzTbzExV.js'), "./lang/ta.json": () => import('./ta-B3m_YUJ4.js'), "./lang/th.json": () => import('./th-dnPspbxQ.js'), "./lang/tr.json": () => import('./tr-Blluguqp.js'), "./lang/uk.json": () => import('./uk-CKgyQx4N.js'), "./lang/ur.json": () => import('./ur-CmiNuJ_u.js'), "./lang/uz.json": () => import('./uz-CKVFBI7d.js'), "./lang/zh-CN.json": () => import('./zh-CN-KerPSYvC.js'), "./lang/zh-TW.json": () => import('./zh-TW-BqHB1GW9.js') });
function is_translation_metadata(obj) {
  const result = obj && typeof obj === "object" && obj.__type__ === "translation_metadata" && typeof obj.key === "string";
  return result;
}
function process_langs() {
  const lazy_langs = Object.fromEntries(
    Object.entries(langs).map(([path, mod]) => [
      path.split("/").pop().split(".")[0],
      { type: "lazy", data: mod }
    ])
  );
  return {
    ...lazy_langs,
    en: { type: "static", data: en }
  };
}
const processed_langs = process_langs();
const available_locales = Object.keys(processed_langs);
const language_choices = Object.entries(
  processed_langs
).map(([code2]) => [lang_map[code2] || code2, code2]);
let i18n_initialized = false;
let previous_translations;
function get_lang_from_preferred_locale(header) {
  const options2 = header.split(",").map(
    (value) => value.includes(";") ? value.split(";").slice(0, 2) : [value, 1]
  );
  options2.sort(
    (a, b) => parseFloat(b[1]) - parseFloat(a[1])
  );
  for (const [lang, _] of options2) {
    if (available_locales.includes(lang)) {
      return lang;
    }
  }
  return null;
}
async function setupi18n(custom_translations, preferred_locale) {
  const should_reinitialize = i18n_initialized && custom_translations !== previous_translations;
  if (i18n_initialized && !should_reinitialize) {
    return;
  }
  const translations_to_use = custom_translations ?? previous_translations ?? {};
  if (custom_translations !== void 0) {
    previous_translations = custom_translations;
  }
  load_translations({
    processed_langs,
    custom_translations: translations_to_use
  });
  let initial_locale = null;
  const browser_locale = getLocaleFromNavigator();
  if (preferred_locale) {
    initial_locale = get_lang_from_preferred_locale(preferred_locale);
  } else {
    initial_locale = browser_locale && available_locales.includes(browser_locale) ? browser_locale : null;
  }
  if (!initial_locale) {
    const normalized_locale = browser_locale?.split("-")[0];
    initial_locale = normalized_locale && available_locales.includes(normalized_locale) ? normalized_locale : "en";
  }
  await init({
    fallbackLocale: "en",
    initialLocale: initial_locale
  });
  i18n_initialized = true;
}
function changeLocale(new_locale) {
  $locale.set(new_locale);
}
function get_initial_locale(browser_locale, available_locales2, fallback_locale = "en") {
  if (!browser_locale) return fallback_locale;
  if (available_locales2.includes(browser_locale)) {
    return browser_locale;
  }
  return fallback_locale;
}
function load_translations(translations) {
  if (!translations) {
    return;
  }
  try {
    for (const lang in translations.custom_translations) {
      addMessages(lang, translations.custom_translations[lang]);
    }
    for (const lang in translations.processed_langs) {
      if (lang === "en" && translations.processed_langs[lang].type === "static") {
        addMessages(lang, en);
      } else if (translations.processed_langs[lang].type === "lazy") {
        registerLocaleLoader(lang, translations.processed_langs[lang].data);
      }
    }
  } catch (e) {
    console.error("Error loading translations:", e);
  }
  for (const lang in loading) {
    addMessages(lang, {
      common: { loading: loading[lang] }
    });
  }
}

let ssr = true;
async function load$1({
  url,
  data: {
    server,
    port,
    local_dev_mode,
    accept_language,
    root_url,
    mount_path,
    cookie,
    auth_required
  },
  route
}) {
  let app;
  const api_url = browser && !local_dev_mode && root_url ? new URL(mount_path || "/", root_url).href : server;
  const deepLink = url.searchParams.get("deep_link");
  const headers = new Headers();
  if (!browser) {
    headers.append("x-gradio-server", root_url);
    if (cookie) {
      headers.append("Cookie", cookie);
    }
  } else {
    headers.append(
      "x-gradio-server",
      new URL(mount_path, location.origin).href
    );
  }
  if (auth_required) {
    await setupi18n(void 0, accept_language);
    return {
      config: {
        auth_required: true,
        auth_message: "",
        components: [],
        current_page: "",
        dependencies: [],
        layout: {},
        pages: [],
        page: {},
        root: root_url,
        space_id: null,
        analytics_enabled: false,
        connect_heartbeat: false,
        css: "",
        js: "",
        theme_hash: 0,
        head: "",
        dev_mode: false,
        enable_queue: false,
        show_error: false,
        fill_height: false,
        fill_width: false,
        mode: "blocks",
        theme: "default",
        title: "",
        version: "",
        api_prefix: "",
        is_space: false,
        is_colab: false,
        footer_links: ["gradio", "settings"],
        stylesheets: [],
        protocol: "sse_v3",
        username: ""
      },
      api_url,
      layout: {},
      app: null
    };
  }
  try {
    app = await Client.connect(api_url, {
      with_null_state: true,
      events: ["data", "log", "status", "render"],
      query_params: deepLink ? { deep_link: deepLink } : void 0,
      headers,
      cookies: cookie || void 0
    });
  } catch (error) {
    const error_message = error.message || "";
    let auth_message = "";
    if (!error_message.includes(MISSING_CREDENTIALS_MSG)) {
      auth_message = error_message.replace(/^Error:?\s*/, "");
    }
    await setupi18n(void 0, accept_language);
    return {
      config: {
        auth_message,
        auth_required: true,
        components: [],
        current_page: "",
        dependencies: [],
        layout: {},
        pages: [],
        page: {},
        root: root_url,
        space_id: null,
        analytics_enabled: false,
        connect_heartbeat: false,
        css: "",
        js: "",
        theme_hash: 0,
        head: "",
        dev_mode: false,
        enable_queue: false,
        show_error: false,
        fill_height: false,
        fill_width: false,
        mode: "blocks",
        theme: "default",
        title: "",
        version: "",
        api_prefix: "",
        is_space: false,
        is_colab: false,
        footer_links: ["gradio", "settings"],
        stylesheets: [],
        protocol: "sse_v3",
        username: ""
      },
      api_url,
      layout: {},
      app: null
    };
  }
  if (!app.config) {
    throw new Error("No config found");
  }
  let page_config = app.get_url_config(url.toString());
  await setupi18n(app.config?.i18n_translations || void 0, accept_language);
  return {
    config: page_config,
    api_url,
    app
  };
}

var _page_ts = /*#__PURE__*/Object.freeze({
  __proto__: null,
  load: load$1,
  ssr: ssr
});

async function load({ request }) {
  const server = request.headers.get("x-gradio-server") || "http://127.0.0.1:7860";
  const port = request.headers.get("x-gradio-port") || "7860";
  const local_dev_mode = request.headers.get("x-gradio-local-dev-mode") || dev ? "true" : void 0;
  const accept_language = request.headers.get("accept-language") || "en";
  const mount_path = request.headers.get("x-gradio-mounted-path") || "/";
  const real_url = new URL(
    request.headers.get("x-gradio-original-url") || server
  ).origin;
  const cookie = request.headers.get("cookie");
  let auth_required = false;
  try {
    const configResponse = await fetch(`${server}/config`, {
      headers: {
        ...cookie ? { Cookie: cookie } : {}
      }
    });
    if (configResponse.status === 401) {
      auth_required = true;
    }
  } catch (e) {
  }
  const root_url = new URL(mount_path, real_url).href.replace(/\/$/, "");
  return {
    server,
    root_url,
    mount_path,
    port,
    local_dev_mode,
    accept_language,
    cookie,
    auth_required
  };
}

var _page_server_ts = /*#__PURE__*/Object.freeze({
  __proto__: null,
  load: load
});

const index = 2;
let component_cache;
const component = async () => component_cache ??= (await import('./_page.svelte-pMWlccma.js')).default;
const universal_id = "src/routes/[...catchall]/+page.ts";
const server_id = "src/routes/[...catchall]/+page.server.ts";
const imports = ["_app/immutable/nodes/2.DUizLbIX.js","_app/immutable/chunks/o-Ck1HYF.js","_app/immutable/chunks/Db2VRVPO.js","_app/immutable/chunks/DcfTJGrh.js","_app/immutable/chunks/43y08qZT.js","_app/immutable/chunks/C903eyvM.js","_app/immutable/chunks/BEYfkQLb.js","_app/immutable/chunks/DE3DmrRd.js","_app/immutable/chunks/DBxpX0Ro.js","_app/immutable/chunks/DA4iNbiz.js","_app/immutable/chunks/1t7wABRY.js"];
const stylesheets = ["_app/immutable/assets/2.PJ8HYwhy.css","_app/immutable/assets/Index.CVv5Ke0n.css","_app/immutable/assets/Index.KYyPl9cR.css","_app/immutable/assets/Example.DQklQyrP.css","_app/immutable/assets/Upload.B9AbOy1B.css","_app/immutable/assets/index.MzKMzM4L.css","_app/immutable/assets/MarkdownCode.BFC2oRGU.css","_app/immutable/assets/Index.uA5k4uH0.css","_app/immutable/assets/Example.bEuOGW3l.css","_app/immutable/assets/Checkbox.CJBAs2gn.css","_app/immutable/assets/Example.Bq3fAzQx.css","_app/immutable/assets/Index.Cd1I_K0a.css","_app/immutable/assets/Example.ZgyKzkPU.css","_app/immutable/assets/Code.DSw5753G.css","_app/immutable/assets/Example.BpLG3o2o.css","_app/immutable/assets/Index.CbmayjvD.css","_app/immutable/assets/Example.B0qjHn8W.css","_app/immutable/assets/Index.-t1xH8ZO.css","_app/immutable/assets/Index.9MMRKKEp.css","_app/immutable/assets/Index.1AccPlNI.css","_app/immutable/assets/Example.B67Cdnj0.css","_app/immutable/assets/Dropdown._c0K7TyY.css","_app/immutable/assets/Index.v1iUT5Sq.css","_app/immutable/assets/Example.ml6sVHU8.css","_app/immutable/assets/Index.dUn14qUu.css","_app/immutable/assets/Index.CR7_W0IB.css","_app/immutable/assets/Example.CHp_DfzD.css","_app/immutable/assets/FileUpload.DhzRhTM7.css","_app/immutable/assets/Example.Bm-XFf-B.css","_app/immutable/assets/Index.CL7x-fxw.css","_app/immutable/assets/Example.Cq74O62D.css","_app/immutable/assets/ImageUploader.79Z_jqCu.css","_app/immutable/assets/Index.BlyT2RM2.css","_app/immutable/assets/Example.BDoLBXEz.css","_app/immutable/assets/Index.ipOE50af.css","_app/immutable/assets/Index.DxNvd7lC.css","_app/immutable/assets/HTML.Czz-mkVb.css","_app/immutable/assets/Example.1AHPPrE3.css","_app/immutable/assets/Index.Bdoky1Bb.css","_app/immutable/assets/Example.b2fpUOu4.css","_app/immutable/assets/Example.YdVSiIuW.css","_app/immutable/assets/Index.BebY3ZUz.css","_app/immutable/assets/JSON.DCdmtE2W.css","_app/immutable/assets/Example.BmdzPZC6.css","_app/immutable/assets/Index.CwrN3-r3.css","_app/immutable/assets/Example.CNix_cQb.css","_app/immutable/assets/Index.BEAewkRY.css","_app/immutable/assets/Model3D.C7UzhPnn.css","_app/immutable/assets/Example.BcEr-euO.css","_app/immutable/assets/Index.amT-rIdI.css","_app/immutable/assets/Example.DCuDV3ni.css","_app/immutable/assets/Index.BcxwxuuF.css","_app/immutable/assets/Index.qjglRoiX.css","_app/immutable/assets/Example.CilajK2g.css","_app/immutable/assets/Index.D6Y3g9po.css","_app/immutable/assets/Example.D6QAAQGU.css","_app/immutable/assets/Index.CNeZGhVI.css","_app/immutable/assets/Example.l4VocDuK.css","_app/immutable/assets/Index.BiXBvr9o.css","_app/immutable/assets/Index.BnASDwkz.css","_app/immutable/assets/Index.Dhk15SNf.css","_app/immutable/assets/Example.DEZCrt3g.css","_app/immutable/assets/Index.Dk23IKH5.css","_app/immutable/assets/Walkthrough.CTImcoqj.css","_app/immutable/assets/Index.D2fA4viL.css","_app/immutable/assets/Index.DCXyjWeM.css","_app/immutable/assets/Index.Dm9eNXb9.css","_app/immutable/assets/Example.Bqpu_FoT.css","_app/immutable/assets/index.BcUNcoUa.css","_app/immutable/assets/Example.mgLMZyqj.css","_app/immutable/assets/Index.DOcbyfJN.css"];
const fonts = [];

var _2 = /*#__PURE__*/Object.freeze({
  __proto__: null,
  component: component,
  fonts: fonts,
  imports: imports,
  index: index,
  server: _page_server_ts,
  server_id: server_id,
  stylesheets: stylesheets,
  universal: _page_ts,
  universal_id: universal_id
});

export { $format as $, AsyncFunction as A, FileData as F, _2 as _, prepare_files as a, $locale as b, commonjsGlobal as c, get_inputs_outputs as d, get_component as e, determine_interactivity as f, getDefaultExportFromCjs as g, changeLocale as h, get_initial_locale as i, is_translation_metadata as j, load_translations as k, language_choices as l, mount_css as m, navbar_config as n, process_langs as o, prefix_css as p, requireCjs as r, setupi18n as s };
//# sourceMappingURL=2-BAeILJ1g.js.map
