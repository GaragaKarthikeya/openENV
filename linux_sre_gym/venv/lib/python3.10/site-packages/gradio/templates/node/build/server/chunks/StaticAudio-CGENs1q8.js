import './async-D55cHugf.js';
import { d as bind_props, f as attr_class, a as attr, g as attr_style, i as stringify } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { u as uploadToHuggingFace, f as format_time } from './utils.svelte-aYomt3KL.js';
import { B as BlockLabel } from './BlockLabel-XZM1LqL4.js';
import { D as DownloadLink } from './DownloadLink-BMNxSkQ8.js';
import { I as IconButton } from './IconButton-D82v5nCM.js';
import { E as Empty } from './Empty-BHvMDNv5.js';
import { S as ShareButton } from './ShareButton-DnAPtcgb.js';
import { D as Download } from './Download-ByiErn53.js';
import { M as Music } from './Music-DcoyPX64.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import { M as MinimalAudioPlayer } from './MinimalAudioPlayer-Bn_OZwMw.js';
import { V as VolumeLevels, P as Pause, T as Trim } from './VolumeLevels-C0Xso12f.js';
import { P as Play } from './Play-BlZWjD1q.js';
import { U as Undo } from './Undo-Col2BcUY.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './index-server-CQz6EZl_.js';

function Backward($$renderer) {
  $$renderer.push(`<svg xmlns="http://www.w3.org/2000/svg" width="24px" height="24px" fill="currentColor" stroke-width="1.5" viewBox="0 0 24 24" color="currentColor"><path stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" d="M21.044 5.704a.6.6 0 0 1 .956.483v11.626a.6.6 0 0 1-.956.483l-7.889-5.813a.6.6 0 0 1 0-.966l7.89-5.813ZM10.044 5.704a.6.6 0 0 1 .956.483v11.626a.6.6 0 0 1-.956.483l-7.888-5.813a.6.6 0 0 1 0-.966l7.888-5.813Z"></path></svg>`);
}
function ClosedCaption($$renderer) {
  $$renderer.push(`<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" class="feather feather-closed-caption"><rect x="2" y="6" width="20" height="12" rx="2" ry="2"></rect><text x="12" y="15" font-family="sans-serif" font-size="8" font-weight="bold" fill="currentColor" stroke="none" text-anchor="middle">CC</text></svg>`);
}
function Forward($$renderer) {
  $$renderer.push(`<svg xmlns="http://www.w3.org/2000/svg" width="24px" height="24px" fill="currentColor" stroke-width="1.5" viewBox="0 0 24 24" color="currentColor"><path stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" d="M2.956 5.704A.6.6 0 0 0 2 6.187v11.626a.6.6 0 0 0 .956.483l7.889-5.813a.6.6 0 0 0 0-.966l-7.89-5.813ZM13.956 5.704a.6.6 0 0 0-.956.483v11.626a.6.6 0 0 0 .956.483l7.889-5.813a.6.6 0 0 0 0-.966l-7.89-5.813Z"></path></svg>`);
}
const get_skip_rewind_amount = (audio_duration, skip_length) => {
  if (!skip_length) {
    skip_length = 5;
  }
  return audio_duration / 100 * skip_length || 5;
};
function VolumeControl($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      currentVolume = void 0,
      show_volume_slider = void 0,
      waveform
    } = $$props;
    $$renderer2.push(`<input id="volume" class="volume-slider svelte-c0ird4" type="range" min="0" max="1" step="0.01"${attr("value", currentVolume)}/>`);
    bind_props($$props, { currentVolume, show_volume_slider });
  });
}
function WaveformControls($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      waveform,
      audio_duration,
      i18n,
      playing,
      show_redo = false,
      interactive = false,
      handle_trim_audio,
      mode = void 0,
      container,
      handle_reset_value,
      waveform_options = {},
      trim_region_settings = {},
      show_volume_slider = void 0,
      editable = true,
      subtitles_toggle = void 0,
      show_subtitles = false,
      trimDuration = void 0
    } = $$props;
    let playbackSpeeds = [0.5, 1, 1.5, 2];
    let playbackSpeed = playbackSpeeds[1];
    let currentVolume = 1;
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      $$renderer3.push(`<div class="controls svelte-72dh9g" data-testid="waveform-controls"><div class="control-wrapper svelte-72dh9g"><button class="action icon volume svelte-72dh9g" aria-label="Adjust volume"${attr_style("", {
        color: show_volume_slider ? "var(--color-accent)" : "var(--neutral-400)"
      })}>`);
      VolumeLevels($$renderer3, { currentVolume });
      $$renderer3.push(`<!----></button> `);
      if (show_volume_slider) {
        $$renderer3.push("<!--[-->");
        VolumeControl($$renderer3, {
          waveform,
          get currentVolume() {
            return currentVolume;
          },
          set currentVolume($$value) {
            currentVolume = $$value;
            $$settled = false;
          },
          get show_volume_slider() {
            return show_volume_slider;
          },
          set show_volume_slider($$value) {
            show_volume_slider = $$value;
            $$settled = false;
          }
        });
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> <button${attr_class("playback icon svelte-72dh9g", void 0, { "hidden": show_volume_slider })}${attr("aria-label", `Adjust playback speed to ${playbackSpeeds[(playbackSpeeds.indexOf(playbackSpeed) + 1) % playbackSpeeds.length]}x`)}><span>${escape_html(playbackSpeed)}x</span></button></div> <div class="play-pause-wrapper svelte-72dh9g"><button class="rewind icon svelte-72dh9g"${attr("aria-label", `Skip backwards by ${get_skip_rewind_amount(audio_duration, waveform_options.skip_length)} seconds`)}>`);
      Backward($$renderer3);
      $$renderer3.push(`<!----></button> <button class="play-pause-button icon svelte-72dh9g"${attr("aria-label", playing ? i18n("audio.pause") : i18n("audio.play"))}>`);
      if (playing) {
        $$renderer3.push("<!--[-->");
        Pause($$renderer3);
      } else {
        $$renderer3.push("<!--[!-->");
        Play($$renderer3);
      }
      $$renderer3.push(`<!--]--></button> <button class="skip icon svelte-72dh9g"${attr("aria-label", `Skip forward by ${stringify(get_skip_rewind_amount(audio_duration, waveform_options.skip_length))} seconds`)}>`);
      Forward($$renderer3);
      $$renderer3.push(`<!----></button></div> <div class="settings-wrapper svelte-72dh9g">`);
      if (show_subtitles) {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<button class="action icon cc-button svelte-72dh9g" data-testid="subtitles-toggle"${attr_style(`color: ${stringify(subtitles_toggle ? "var(--color-accent)" : "var(--neutral-400)")}`)}>`);
        ClosedCaption($$renderer3);
        $$renderer3.push(`<!----></button>`);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> `);
      if (editable && interactive) {
        $$renderer3.push("<!--[-->");
        if (show_redo && !mode) {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<button class="action icon svelte-72dh9g" aria-label="Reset audio">`);
          Undo($$renderer3);
          $$renderer3.push(`<!----></button>`);
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> `);
        if (!mode) {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<button class="action icon svelte-72dh9g" aria-label="Trim audio to selection">`);
          Trim($$renderer3);
          $$renderer3.push(`<!----></button>`);
        } else {
          $$renderer3.push("<!--[!-->");
          $$renderer3.push(`<button class="text-button svelte-72dh9g">Trim</button> <button class="text-button svelte-72dh9g">Cancel</button>`);
        }
        $$renderer3.push(`<!--]-->`);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--></div></div>`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { mode, show_volume_slider, subtitles_toggle, trimDuration });
  });
}
function getDefaultExportFromCjs(x) {
  return x && x.__esModule && Object.prototype.hasOwnProperty.call(x, "default") ? x["default"] : x;
}
var urlToolkit = { exports: {} };
(function(module, exports$1) {
  (function(root) {
    var URL_REGEX = /^(?=((?:[a-zA-Z0-9+\-.]+:)?))\1(?=((?:\/\/[^\/?#]*)?))\2(?=((?:(?:[^?#\/]*\/)*[^;?#\/]*)?))\3((?:;[^?#]*)?)(\?[^#]*)?(#[^]*)?$/;
    var FIRST_SEGMENT_REGEX = /^(?=([^\/?#]*))\1([^]*)$/;
    var SLASH_DOT_REGEX = /(?:\/|^)\.(?=\/)/g;
    var SLASH_DOT_DOT_REGEX = /(?:\/|^)\.\.\/(?!\.\.\/)[^\/]*(?=\/)/g;
    var URLToolkit = {
      // If opts.alwaysNormalize is true then the path will always be normalized even when it starts with / or //
      // E.g
      // With opts.alwaysNormalize = false (default, spec compliant)
      // http://a.com/b/cd + /e/f/../g => http://a.com/e/f/../g
      // With opts.alwaysNormalize = true (not spec compliant)
      // http://a.com/b/cd + /e/f/../g => http://a.com/e/g
      buildAbsoluteURL: function(baseURL, relativeURL, opts) {
        opts = opts || {};
        baseURL = baseURL.trim();
        relativeURL = relativeURL.trim();
        if (!relativeURL) {
          if (!opts.alwaysNormalize) {
            return baseURL;
          }
          var basePartsForNormalise = URLToolkit.parseURL(baseURL);
          if (!basePartsForNormalise) {
            throw new Error("Error trying to parse base URL.");
          }
          basePartsForNormalise.path = URLToolkit.normalizePath(
            basePartsForNormalise.path
          );
          return URLToolkit.buildURLFromParts(basePartsForNormalise);
        }
        var relativeParts = URLToolkit.parseURL(relativeURL);
        if (!relativeParts) {
          throw new Error("Error trying to parse relative URL.");
        }
        if (relativeParts.scheme) {
          if (!opts.alwaysNormalize) {
            return relativeURL;
          }
          relativeParts.path = URLToolkit.normalizePath(relativeParts.path);
          return URLToolkit.buildURLFromParts(relativeParts);
        }
        var baseParts = URLToolkit.parseURL(baseURL);
        if (!baseParts) {
          throw new Error("Error trying to parse base URL.");
        }
        if (!baseParts.netLoc && baseParts.path && baseParts.path[0] !== "/") {
          var pathParts = FIRST_SEGMENT_REGEX.exec(baseParts.path);
          baseParts.netLoc = pathParts[1];
          baseParts.path = pathParts[2];
        }
        if (baseParts.netLoc && !baseParts.path) {
          baseParts.path = "/";
        }
        var builtParts = {
          // 2c) Otherwise, the embedded URL inherits the scheme of
          // the base URL.
          scheme: baseParts.scheme,
          netLoc: relativeParts.netLoc,
          path: null,
          params: relativeParts.params,
          query: relativeParts.query,
          fragment: relativeParts.fragment
        };
        if (!relativeParts.netLoc) {
          builtParts.netLoc = baseParts.netLoc;
          if (relativeParts.path[0] !== "/") {
            if (!relativeParts.path) {
              builtParts.path = baseParts.path;
              if (!relativeParts.params) {
                builtParts.params = baseParts.params;
                if (!relativeParts.query) {
                  builtParts.query = baseParts.query;
                }
              }
            } else {
              var baseURLPath = baseParts.path;
              var newPath = baseURLPath.substring(0, baseURLPath.lastIndexOf("/") + 1) + relativeParts.path;
              builtParts.path = URLToolkit.normalizePath(newPath);
            }
          }
        }
        if (builtParts.path === null) {
          builtParts.path = opts.alwaysNormalize ? URLToolkit.normalizePath(relativeParts.path) : relativeParts.path;
        }
        return URLToolkit.buildURLFromParts(builtParts);
      },
      parseURL: function(url) {
        var parts = URL_REGEX.exec(url);
        if (!parts) {
          return null;
        }
        return {
          scheme: parts[1] || "",
          netLoc: parts[2] || "",
          path: parts[3] || "",
          params: parts[4] || "",
          query: parts[5] || "",
          fragment: parts[6] || ""
        };
      },
      normalizePath: function(path) {
        path = path.split("").reverse().join("").replace(SLASH_DOT_REGEX, "");
        while (path.length !== (path = path.replace(SLASH_DOT_DOT_REGEX, "")).length) {
        }
        return path.split("").reverse().join("");
      },
      buildURLFromParts: function(parts) {
        return parts.scheme + parts.netLoc + parts.path + parts.params + parts.query + parts.fragment;
      }
    };
    module.exports = URLToolkit;
  })();
})(urlToolkit);
function ownKeys(e2, r2) {
  var t2 = Object.keys(e2);
  if (Object.getOwnPropertySymbols) {
    var o2 = Object.getOwnPropertySymbols(e2);
    r2 && (o2 = o2.filter(function(r3) {
      return Object.getOwnPropertyDescriptor(e2, r3).enumerable;
    })), t2.push.apply(t2, o2);
  }
  return t2;
}
function _objectSpread2(e2) {
  for (var r2 = 1; r2 < arguments.length; r2++) {
    var t2 = null != arguments[r2] ? arguments[r2] : {};
    r2 % 2 ? ownKeys(Object(t2), true).forEach(function(r3) {
      _defineProperty(e2, r3, t2[r3]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e2, Object.getOwnPropertyDescriptors(t2)) : ownKeys(Object(t2)).forEach(function(r3) {
      Object.defineProperty(e2, r3, Object.getOwnPropertyDescriptor(t2, r3));
    });
  }
  return e2;
}
function _toPrimitive(t2, r2) {
  if ("object" != typeof t2 || !t2) return t2;
  var e2 = t2[Symbol.toPrimitive];
  if (void 0 !== e2) {
    var i2 = e2.call(t2, r2);
    if ("object" != typeof i2) return i2;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return ("string" === r2 ? String : Number)(t2);
}
function _toPropertyKey(t2) {
  var i2 = _toPrimitive(t2, "string");
  return "symbol" == typeof i2 ? i2 : String(i2);
}
function _defineProperty(obj, key, value) {
  key = _toPropertyKey(key);
  if (key in obj) {
    Object.defineProperty(obj, key, {
      value,
      enumerable: true,
      configurable: true,
      writable: true
    });
  } else {
    obj[key] = value;
  }
  return obj;
}
function _extends() {
  _extends = Object.assign ? Object.assign.bind() : function(target) {
    for (var i2 = 1; i2 < arguments.length; i2++) {
      var source = arguments[i2];
      for (var key in source) {
        if (Object.prototype.hasOwnProperty.call(source, key)) {
          target[key] = source[key];
        }
      }
    }
    return target;
  };
  return _extends.apply(this, arguments);
}
const isFiniteNumber = Number.isFinite || function(value) {
  return typeof value === "number" && isFinite(value);
};
const isSafeInteger = Number.isSafeInteger || function(value) {
  return typeof value === "number" && Math.abs(value) <= MAX_SAFE_INTEGER;
};
const MAX_SAFE_INTEGER = Number.MAX_SAFE_INTEGER || 9007199254740991;
let Events = /* @__PURE__ */ (function(Events2) {
  Events2["MEDIA_ATTACHING"] = "hlsMediaAttaching";
  Events2["MEDIA_ATTACHED"] = "hlsMediaAttached";
  Events2["MEDIA_DETACHING"] = "hlsMediaDetaching";
  Events2["MEDIA_DETACHED"] = "hlsMediaDetached";
  Events2["BUFFER_RESET"] = "hlsBufferReset";
  Events2["BUFFER_CODECS"] = "hlsBufferCodecs";
  Events2["BUFFER_CREATED"] = "hlsBufferCreated";
  Events2["BUFFER_APPENDING"] = "hlsBufferAppending";
  Events2["BUFFER_APPENDED"] = "hlsBufferAppended";
  Events2["BUFFER_EOS"] = "hlsBufferEos";
  Events2["BUFFER_FLUSHING"] = "hlsBufferFlushing";
  Events2["BUFFER_FLUSHED"] = "hlsBufferFlushed";
  Events2["MANIFEST_LOADING"] = "hlsManifestLoading";
  Events2["MANIFEST_LOADED"] = "hlsManifestLoaded";
  Events2["MANIFEST_PARSED"] = "hlsManifestParsed";
  Events2["LEVEL_SWITCHING"] = "hlsLevelSwitching";
  Events2["LEVEL_SWITCHED"] = "hlsLevelSwitched";
  Events2["LEVEL_LOADING"] = "hlsLevelLoading";
  Events2["LEVEL_LOADED"] = "hlsLevelLoaded";
  Events2["LEVEL_UPDATED"] = "hlsLevelUpdated";
  Events2["LEVEL_PTS_UPDATED"] = "hlsLevelPtsUpdated";
  Events2["LEVELS_UPDATED"] = "hlsLevelsUpdated";
  Events2["AUDIO_TRACKS_UPDATED"] = "hlsAudioTracksUpdated";
  Events2["AUDIO_TRACK_SWITCHING"] = "hlsAudioTrackSwitching";
  Events2["AUDIO_TRACK_SWITCHED"] = "hlsAudioTrackSwitched";
  Events2["AUDIO_TRACK_LOADING"] = "hlsAudioTrackLoading";
  Events2["AUDIO_TRACK_LOADED"] = "hlsAudioTrackLoaded";
  Events2["SUBTITLE_TRACKS_UPDATED"] = "hlsSubtitleTracksUpdated";
  Events2["SUBTITLE_TRACKS_CLEARED"] = "hlsSubtitleTracksCleared";
  Events2["SUBTITLE_TRACK_SWITCH"] = "hlsSubtitleTrackSwitch";
  Events2["SUBTITLE_TRACK_LOADING"] = "hlsSubtitleTrackLoading";
  Events2["SUBTITLE_TRACK_LOADED"] = "hlsSubtitleTrackLoaded";
  Events2["SUBTITLE_FRAG_PROCESSED"] = "hlsSubtitleFragProcessed";
  Events2["CUES_PARSED"] = "hlsCuesParsed";
  Events2["NON_NATIVE_TEXT_TRACKS_FOUND"] = "hlsNonNativeTextTracksFound";
  Events2["INIT_PTS_FOUND"] = "hlsInitPtsFound";
  Events2["FRAG_LOADING"] = "hlsFragLoading";
  Events2["FRAG_LOAD_EMERGENCY_ABORTED"] = "hlsFragLoadEmergencyAborted";
  Events2["FRAG_LOADED"] = "hlsFragLoaded";
  Events2["FRAG_DECRYPTED"] = "hlsFragDecrypted";
  Events2["FRAG_PARSING_INIT_SEGMENT"] = "hlsFragParsingInitSegment";
  Events2["FRAG_PARSING_USERDATA"] = "hlsFragParsingUserdata";
  Events2["FRAG_PARSING_METADATA"] = "hlsFragParsingMetadata";
  Events2["FRAG_PARSED"] = "hlsFragParsed";
  Events2["FRAG_BUFFERED"] = "hlsFragBuffered";
  Events2["FRAG_CHANGED"] = "hlsFragChanged";
  Events2["FPS_DROP"] = "hlsFpsDrop";
  Events2["FPS_DROP_LEVEL_CAPPING"] = "hlsFpsDropLevelCapping";
  Events2["MAX_AUTO_LEVEL_UPDATED"] = "hlsMaxAutoLevelUpdated";
  Events2["ERROR"] = "hlsError";
  Events2["DESTROYING"] = "hlsDestroying";
  Events2["KEY_LOADING"] = "hlsKeyLoading";
  Events2["KEY_LOADED"] = "hlsKeyLoaded";
  Events2["LIVE_BACK_BUFFER_REACHED"] = "hlsLiveBackBufferReached";
  Events2["BACK_BUFFER_REACHED"] = "hlsBackBufferReached";
  Events2["STEERING_MANIFEST_LOADED"] = "hlsSteeringManifestLoaded";
  return Events2;
})({});
let ErrorTypes = /* @__PURE__ */ (function(ErrorTypes2) {
  ErrorTypes2["NETWORK_ERROR"] = "networkError";
  ErrorTypes2["MEDIA_ERROR"] = "mediaError";
  ErrorTypes2["KEY_SYSTEM_ERROR"] = "keySystemError";
  ErrorTypes2["MUX_ERROR"] = "muxError";
  ErrorTypes2["OTHER_ERROR"] = "otherError";
  return ErrorTypes2;
})({});
let ErrorDetails = /* @__PURE__ */ (function(ErrorDetails2) {
  ErrorDetails2["KEY_SYSTEM_NO_KEYS"] = "keySystemNoKeys";
  ErrorDetails2["KEY_SYSTEM_NO_ACCESS"] = "keySystemNoAccess";
  ErrorDetails2["KEY_SYSTEM_NO_SESSION"] = "keySystemNoSession";
  ErrorDetails2["KEY_SYSTEM_NO_CONFIGURED_LICENSE"] = "keySystemNoConfiguredLicense";
  ErrorDetails2["KEY_SYSTEM_LICENSE_REQUEST_FAILED"] = "keySystemLicenseRequestFailed";
  ErrorDetails2["KEY_SYSTEM_SERVER_CERTIFICATE_REQUEST_FAILED"] = "keySystemServerCertificateRequestFailed";
  ErrorDetails2["KEY_SYSTEM_SERVER_CERTIFICATE_UPDATE_FAILED"] = "keySystemServerCertificateUpdateFailed";
  ErrorDetails2["KEY_SYSTEM_SESSION_UPDATE_FAILED"] = "keySystemSessionUpdateFailed";
  ErrorDetails2["KEY_SYSTEM_STATUS_OUTPUT_RESTRICTED"] = "keySystemStatusOutputRestricted";
  ErrorDetails2["KEY_SYSTEM_STATUS_INTERNAL_ERROR"] = "keySystemStatusInternalError";
  ErrorDetails2["MANIFEST_LOAD_ERROR"] = "manifestLoadError";
  ErrorDetails2["MANIFEST_LOAD_TIMEOUT"] = "manifestLoadTimeOut";
  ErrorDetails2["MANIFEST_PARSING_ERROR"] = "manifestParsingError";
  ErrorDetails2["MANIFEST_INCOMPATIBLE_CODECS_ERROR"] = "manifestIncompatibleCodecsError";
  ErrorDetails2["LEVEL_EMPTY_ERROR"] = "levelEmptyError";
  ErrorDetails2["LEVEL_LOAD_ERROR"] = "levelLoadError";
  ErrorDetails2["LEVEL_LOAD_TIMEOUT"] = "levelLoadTimeOut";
  ErrorDetails2["LEVEL_PARSING_ERROR"] = "levelParsingError";
  ErrorDetails2["LEVEL_SWITCH_ERROR"] = "levelSwitchError";
  ErrorDetails2["AUDIO_TRACK_LOAD_ERROR"] = "audioTrackLoadError";
  ErrorDetails2["AUDIO_TRACK_LOAD_TIMEOUT"] = "audioTrackLoadTimeOut";
  ErrorDetails2["SUBTITLE_LOAD_ERROR"] = "subtitleTrackLoadError";
  ErrorDetails2["SUBTITLE_TRACK_LOAD_TIMEOUT"] = "subtitleTrackLoadTimeOut";
  ErrorDetails2["FRAG_LOAD_ERROR"] = "fragLoadError";
  ErrorDetails2["FRAG_LOAD_TIMEOUT"] = "fragLoadTimeOut";
  ErrorDetails2["FRAG_DECRYPT_ERROR"] = "fragDecryptError";
  ErrorDetails2["FRAG_PARSING_ERROR"] = "fragParsingError";
  ErrorDetails2["FRAG_GAP"] = "fragGap";
  ErrorDetails2["REMUX_ALLOC_ERROR"] = "remuxAllocError";
  ErrorDetails2["KEY_LOAD_ERROR"] = "keyLoadError";
  ErrorDetails2["KEY_LOAD_TIMEOUT"] = "keyLoadTimeOut";
  ErrorDetails2["BUFFER_ADD_CODEC_ERROR"] = "bufferAddCodecError";
  ErrorDetails2["BUFFER_INCOMPATIBLE_CODECS_ERROR"] = "bufferIncompatibleCodecsError";
  ErrorDetails2["BUFFER_APPEND_ERROR"] = "bufferAppendError";
  ErrorDetails2["BUFFER_APPENDING_ERROR"] = "bufferAppendingError";
  ErrorDetails2["BUFFER_STALLED_ERROR"] = "bufferStalledError";
  ErrorDetails2["BUFFER_FULL_ERROR"] = "bufferFullError";
  ErrorDetails2["BUFFER_SEEK_OVER_HOLE"] = "bufferSeekOverHole";
  ErrorDetails2["BUFFER_NUDGE_ON_STALL"] = "bufferNudgeOnStall";
  ErrorDetails2["INTERNAL_EXCEPTION"] = "internalException";
  ErrorDetails2["INTERNAL_ABORTED"] = "aborted";
  ErrorDetails2["UNKNOWN"] = "unknown";
  return ErrorDetails2;
})({});
const noop = function noop2() {
};
const fakeLogger = {
  trace: noop,
  debug: noop,
  log: noop,
  warn: noop,
  info: noop,
  error: noop
};
let exportedLogger = fakeLogger;
const logger = exportedLogger;
const DECIMAL_RESOLUTION_REGEX = /^(\d+)x(\d+)$/;
const ATTR_LIST_REGEX = /(.+?)=(".*?"|.*?)(?:,|$)/g;
class AttrList {
  constructor(attrs) {
    if (typeof attrs === "string") {
      attrs = AttrList.parseAttrList(attrs);
    }
    _extends(this, attrs);
  }
  get clientAttrs() {
    return Object.keys(this).filter((attr) => attr.substring(0, 2) === "X-");
  }
  decimalInteger(attrName) {
    const intValue = parseInt(this[attrName], 10);
    if (intValue > Number.MAX_SAFE_INTEGER) {
      return Infinity;
    }
    return intValue;
  }
  hexadecimalInteger(attrName) {
    if (this[attrName]) {
      let stringValue = (this[attrName] || "0x").slice(2);
      stringValue = (stringValue.length & 1 ? "0" : "") + stringValue;
      const value = new Uint8Array(stringValue.length / 2);
      for (let i2 = 0; i2 < stringValue.length / 2; i2++) {
        value[i2] = parseInt(stringValue.slice(i2 * 2, i2 * 2 + 2), 16);
      }
      return value;
    } else {
      return null;
    }
  }
  hexadecimalIntegerAsNumber(attrName) {
    const intValue = parseInt(this[attrName], 16);
    if (intValue > Number.MAX_SAFE_INTEGER) {
      return Infinity;
    }
    return intValue;
  }
  decimalFloatingPoint(attrName) {
    return parseFloat(this[attrName]);
  }
  optionalFloat(attrName, defaultValue) {
    const value = this[attrName];
    return value ? parseFloat(value) : defaultValue;
  }
  enumeratedString(attrName) {
    return this[attrName];
  }
  bool(attrName) {
    return this[attrName] === "YES";
  }
  decimalResolution(attrName) {
    const res = DECIMAL_RESOLUTION_REGEX.exec(this[attrName]);
    if (res === null) {
      return void 0;
    }
    return {
      width: parseInt(res[1], 10),
      height: parseInt(res[2], 10)
    };
  }
  static parseAttrList(input) {
    let match;
    const attrs = {};
    const quote = '"';
    ATTR_LIST_REGEX.lastIndex = 0;
    while ((match = ATTR_LIST_REGEX.exec(input)) !== null) {
      let value = match[2];
      if (value.indexOf(quote) === 0 && value.lastIndexOf(quote) === value.length - 1) {
        value = value.slice(1, -1);
      }
      const name = match[1].trim();
      attrs[name] = value;
    }
    return attrs;
  }
}
class DateRange {
  constructor(dateRangeAttr, dateRangeWithSameId) {
    this.attr = void 0;
    this._startDate = void 0;
    this._endDate = void 0;
    this._badValueForSameId = void 0;
    if (dateRangeWithSameId) {
      const previousAttr = dateRangeWithSameId.attr;
      for (const key in previousAttr) {
        if (Object.prototype.hasOwnProperty.call(dateRangeAttr, key) && dateRangeAttr[key] !== previousAttr[key]) {
          logger.warn(`DATERANGE tag attribute: "${key}" does not match for tags with ID: "${dateRangeAttr.ID}"`);
          this._badValueForSameId = key;
          break;
        }
      }
      dateRangeAttr = _extends(new AttrList({}), previousAttr, dateRangeAttr);
    }
    this.attr = dateRangeAttr;
    this._startDate = new Date(dateRangeAttr["START-DATE"]);
    if ("END-DATE" in this.attr) {
      const endDate = new Date(this.attr["END-DATE"]);
      if (isFiniteNumber(endDate.getTime())) {
        this._endDate = endDate;
      }
    }
  }
  get id() {
    return this.attr.ID;
  }
  get class() {
    return this.attr.CLASS;
  }
  get startDate() {
    return this._startDate;
  }
  get endDate() {
    if (this._endDate) {
      return this._endDate;
    }
    const duration = this.duration;
    if (duration !== null) {
      return new Date(this._startDate.getTime() + duration * 1e3);
    }
    return null;
  }
  get duration() {
    if ("DURATION" in this.attr) {
      const duration = this.attr.decimalFloatingPoint("DURATION");
      if (isFiniteNumber(duration)) {
        return duration;
      }
    } else if (this._endDate) {
      return (this._endDate.getTime() - this._startDate.getTime()) / 1e3;
    }
    return null;
  }
  get plannedDuration() {
    if ("PLANNED-DURATION" in this.attr) {
      return this.attr.decimalFloatingPoint("PLANNED-DURATION");
    }
    return null;
  }
  get endOnNext() {
    return this.attr.bool("END-ON-NEXT");
  }
  get isValid() {
    return !!this.id && !this._badValueForSameId && isFiniteNumber(this.startDate.getTime()) && (this.duration === null || this.duration >= 0) && (!this.endOnNext || !!this.class);
  }
}
class LoadStats {
  constructor() {
    this.aborted = false;
    this.loaded = 0;
    this.retry = 0;
    this.total = 0;
    this.chunkCount = 0;
    this.bwEstimate = 0;
    this.loading = {
      start: 0,
      first: 0,
      end: 0
    };
    this.parsing = {
      start: 0,
      end: 0
    };
    this.buffering = {
      start: 0,
      first: 0,
      end: 0
    };
  }
}
var ElementaryStreamTypes = {
  AUDIO: "audio",
  VIDEO: "video",
  AUDIOVIDEO: "audiovideo"
};
function base64Decode(base64encodedStr) {
  return Uint8Array.from(atob(base64encodedStr), (c) => c.charCodeAt(0));
}
function getKeyIdBytes(str) {
  const keyIdbytes = strToUtf8array(str).subarray(0, 16);
  const paddedkeyIdbytes = new Uint8Array(16);
  paddedkeyIdbytes.set(keyIdbytes, 16 - keyIdbytes.length);
  return paddedkeyIdbytes;
}
function changeEndianness(keyId) {
  const swap = function swap2(array, from, to) {
    const cur = array[from];
    array[from] = array[to];
    array[to] = cur;
  };
  swap(keyId, 0, 3);
  swap(keyId, 1, 2);
  swap(keyId, 4, 5);
  swap(keyId, 6, 7);
}
function convertDataUriToArrayBytes(uri) {
  const colonsplit = uri.split(":");
  let keydata = null;
  if (colonsplit[0] === "data" && colonsplit.length === 2) {
    const semicolonsplit = colonsplit[1].split(";");
    const commasplit = semicolonsplit[semicolonsplit.length - 1].split(",");
    if (commasplit.length === 2) {
      const isbase64 = commasplit[0] === "base64";
      const data = commasplit[1];
      if (isbase64) {
        semicolonsplit.splice(-1, 1);
        keydata = base64Decode(data);
      } else {
        keydata = getKeyIdBytes(data);
      }
    }
  }
  return keydata;
}
function strToUtf8array(str) {
  return Uint8Array.from(unescape(encodeURIComponent(str)), (c) => c.charCodeAt(0));
}
const optionalSelf = typeof self !== "undefined" ? self : void 0;
var KeySystems = {
  CLEARKEY: "org.w3.clearkey",
  FAIRPLAY: "com.apple.fps",
  PLAYREADY: "com.microsoft.playready",
  WIDEVINE: "com.widevine.alpha"
};
var KeySystemFormats = {
  CLEARKEY: "org.w3.clearkey",
  FAIRPLAY: "com.apple.streamingkeydelivery",
  PLAYREADY: "com.microsoft.playready",
  WIDEVINE: "urn:uuid:edef8ba9-79d6-4ace-a3c8-27dcd51d21ed"
};
function keySystemFormatToKeySystemDomain(format2) {
  switch (format2) {
    case KeySystemFormats.FAIRPLAY:
      return KeySystems.FAIRPLAY;
    case KeySystemFormats.PLAYREADY:
      return KeySystems.PLAYREADY;
    case KeySystemFormats.WIDEVINE:
      return KeySystems.WIDEVINE;
    case KeySystemFormats.CLEARKEY:
      return KeySystems.CLEARKEY;
  }
}
var KeySystemIds = {
  WIDEVINE: "edef8ba979d64acea3c827dcd51d21ed"
};
function keySystemIdToKeySystemDomain(systemId) {
  if (systemId === KeySystemIds.WIDEVINE) {
    return KeySystems.WIDEVINE;
  }
}
function keySystemDomainToKeySystemFormat(keySystem) {
  switch (keySystem) {
    case KeySystems.FAIRPLAY:
      return KeySystemFormats.FAIRPLAY;
    case KeySystems.PLAYREADY:
      return KeySystemFormats.PLAYREADY;
    case KeySystems.WIDEVINE:
      return KeySystemFormats.WIDEVINE;
    case KeySystems.CLEARKEY:
      return KeySystemFormats.CLEARKEY;
  }
}
function getKeySystemsForConfig(config) {
  const {
    drmSystems,
    widevineLicenseUrl
  } = config;
  const keySystemsToAttempt = drmSystems ? [KeySystems.FAIRPLAY, KeySystems.WIDEVINE, KeySystems.PLAYREADY, KeySystems.CLEARKEY].filter((keySystem) => !!drmSystems[keySystem]) : [];
  if (!keySystemsToAttempt[KeySystems.WIDEVINE] && widevineLicenseUrl) {
    keySystemsToAttempt.push(KeySystems.WIDEVINE);
  }
  return keySystemsToAttempt;
}
const requestMediaKeySystemAccess = (function(_optionalSelf$navigat) {
  if (optionalSelf != null && (_optionalSelf$navigat = optionalSelf.navigator) != null && _optionalSelf$navigat.requestMediaKeySystemAccess) {
    return self.navigator.requestMediaKeySystemAccess.bind(self.navigator);
  } else {
    return null;
  }
})();
function getSupportedMediaKeySystemConfigurations(keySystem, audioCodecs, videoCodecs, drmSystemOptions) {
  let initDataTypes;
  switch (keySystem) {
    case KeySystems.FAIRPLAY:
      initDataTypes = ["cenc", "sinf"];
      break;
    case KeySystems.WIDEVINE:
    case KeySystems.PLAYREADY:
      initDataTypes = ["cenc"];
      break;
    case KeySystems.CLEARKEY:
      initDataTypes = ["cenc", "keyids"];
      break;
    default:
      throw new Error(`Unknown key-system: ${keySystem}`);
  }
  return createMediaKeySystemConfigurations(initDataTypes, audioCodecs, videoCodecs, drmSystemOptions);
}
function createMediaKeySystemConfigurations(initDataTypes, audioCodecs, videoCodecs, drmSystemOptions) {
  const baseConfig = {
    initDataTypes,
    persistentState: drmSystemOptions.persistentState || "optional",
    distinctiveIdentifier: drmSystemOptions.distinctiveIdentifier || "optional",
    sessionTypes: drmSystemOptions.sessionTypes || [drmSystemOptions.sessionType || "temporary"],
    audioCapabilities: audioCodecs.map((codec) => ({
      contentType: `audio/mp4; codecs="${codec}"`,
      robustness: drmSystemOptions.audioRobustness || "",
      encryptionScheme: drmSystemOptions.audioEncryptionScheme || null
    })),
    videoCapabilities: videoCodecs.map((codec) => ({
      contentType: `video/mp4; codecs="${codec}"`,
      robustness: drmSystemOptions.videoRobustness || "",
      encryptionScheme: drmSystemOptions.videoEncryptionScheme || null
    }))
  };
  return [baseConfig];
}
function sliceUint8(array, start, end) {
  return Uint8Array.prototype.slice ? array.slice(start, end) : new Uint8Array(Array.prototype.slice.call(array, start, end));
}
const isHeader$2 = (data, offset) => {
  if (offset + 10 <= data.length) {
    if (data[offset] === 73 && data[offset + 1] === 68 && data[offset + 2] === 51) {
      if (data[offset + 3] < 255 && data[offset + 4] < 255) {
        if (data[offset + 6] < 128 && data[offset + 7] < 128 && data[offset + 8] < 128 && data[offset + 9] < 128) {
          return true;
        }
      }
    }
  }
  return false;
};
const isFooter = (data, offset) => {
  if (offset + 10 <= data.length) {
    if (data[offset] === 51 && data[offset + 1] === 68 && data[offset + 2] === 73) {
      if (data[offset + 3] < 255 && data[offset + 4] < 255) {
        if (data[offset + 6] < 128 && data[offset + 7] < 128 && data[offset + 8] < 128 && data[offset + 9] < 128) {
          return true;
        }
      }
    }
  }
  return false;
};
const getID3Data = (data, offset) => {
  const front = offset;
  let length = 0;
  while (isHeader$2(data, offset)) {
    length += 10;
    const size = readSize(data, offset + 6);
    length += size;
    if (isFooter(data, offset + 10)) {
      length += 10;
    }
    offset += length;
  }
  if (length > 0) {
    return data.subarray(front, front + length);
  }
  return void 0;
};
const readSize = (data, offset) => {
  let size = 0;
  size = (data[offset] & 127) << 21;
  size |= (data[offset + 1] & 127) << 14;
  size |= (data[offset + 2] & 127) << 7;
  size |= data[offset + 3] & 127;
  return size;
};
const canParse$2 = (data, offset) => {
  return isHeader$2(data, offset) && readSize(data, offset + 6) + 10 <= data.length - offset;
};
const getTimeStamp = (data) => {
  const frames = getID3Frames(data);
  for (let i2 = 0; i2 < frames.length; i2++) {
    const frame = frames[i2];
    if (isTimeStampFrame(frame)) {
      return readTimeStamp(frame);
    }
  }
  return void 0;
};
const isTimeStampFrame = (frame) => {
  return frame && frame.key === "PRIV" && frame.info === "com.apple.streaming.transportStreamTimestamp";
};
const getFrameData = (data) => {
  const type = String.fromCharCode(data[0], data[1], data[2], data[3]);
  const size = readSize(data, 4);
  const offset = 10;
  return {
    type,
    size,
    data: data.subarray(offset, offset + size)
  };
};
const getID3Frames = (id3Data) => {
  let offset = 0;
  const frames = [];
  while (isHeader$2(id3Data, offset)) {
    const size = readSize(id3Data, offset + 6);
    offset += 10;
    const end = offset + size;
    while (offset + 8 < end) {
      const frameData = getFrameData(id3Data.subarray(offset));
      const frame = decodeFrame(frameData);
      if (frame) {
        frames.push(frame);
      }
      offset += frameData.size + 10;
    }
    if (isFooter(id3Data, offset)) {
      offset += 10;
    }
  }
  return frames;
};
const decodeFrame = (frame) => {
  if (frame.type === "PRIV") {
    return decodePrivFrame(frame);
  } else if (frame.type[0] === "W") {
    return decodeURLFrame(frame);
  }
  return decodeTextFrame(frame);
};
const decodePrivFrame = (frame) => {
  if (frame.size < 2) {
    return void 0;
  }
  const owner = utf8ArrayToStr(frame.data, true);
  const privateData = new Uint8Array(frame.data.subarray(owner.length + 1));
  return {
    key: frame.type,
    info: owner,
    data: privateData.buffer
  };
};
const decodeTextFrame = (frame) => {
  if (frame.size < 2) {
    return void 0;
  }
  if (frame.type === "TXXX") {
    let index = 1;
    const description = utf8ArrayToStr(frame.data.subarray(index), true);
    index += description.length + 1;
    const value = utf8ArrayToStr(frame.data.subarray(index));
    return {
      key: frame.type,
      info: description,
      data: value
    };
  }
  const text = utf8ArrayToStr(frame.data.subarray(1));
  return {
    key: frame.type,
    data: text
  };
};
const decodeURLFrame = (frame) => {
  if (frame.type === "WXXX") {
    if (frame.size < 2) {
      return void 0;
    }
    let index = 1;
    const description = utf8ArrayToStr(frame.data.subarray(index), true);
    index += description.length + 1;
    const value = utf8ArrayToStr(frame.data.subarray(index));
    return {
      key: frame.type,
      info: description,
      data: value
    };
  }
  const url = utf8ArrayToStr(frame.data);
  return {
    key: frame.type,
    data: url
  };
};
const readTimeStamp = (timeStampFrame) => {
  if (timeStampFrame.data.byteLength === 8) {
    const data = new Uint8Array(timeStampFrame.data);
    const pts33Bit = data[3] & 1;
    let timestamp = (data[4] << 23) + (data[5] << 15) + (data[6] << 7) + data[7];
    timestamp /= 45;
    if (pts33Bit) {
      timestamp += 4772185884e-2;
    }
    return Math.round(timestamp);
  }
  return void 0;
};
const utf8ArrayToStr = (array, exitOnNull = false) => {
  const decoder2 = getTextDecoder();
  if (decoder2) {
    const decoded = decoder2.decode(array);
    if (exitOnNull) {
      const idx = decoded.indexOf("\0");
      return idx !== -1 ? decoded.substring(0, idx) : decoded;
    }
    return decoded.replace(/\0/g, "");
  }
  const len = array.length;
  let c;
  let char2;
  let char3;
  let out = "";
  let i2 = 0;
  while (i2 < len) {
    c = array[i2++];
    if (c === 0 && exitOnNull) {
      return out;
    } else if (c === 0 || c === 3) {
      continue;
    }
    switch (c >> 4) {
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
        out += String.fromCharCode(c);
        break;
      case 12:
      case 13:
        char2 = array[i2++];
        out += String.fromCharCode((c & 31) << 6 | char2 & 63);
        break;
      case 14:
        char2 = array[i2++];
        char3 = array[i2++];
        out += String.fromCharCode((c & 15) << 12 | (char2 & 63) << 6 | (char3 & 63) << 0);
        break;
    }
  }
  return out;
};
let decoder;
function getTextDecoder() {
  if (navigator.userAgent.includes("PlayStation 4")) {
    return;
  }
  if (!decoder && typeof self.TextDecoder !== "undefined") {
    decoder = new self.TextDecoder("utf-8");
  }
  return decoder;
}
const Hex = {
  hexDump: function(array) {
    let str = "";
    for (let i2 = 0; i2 < array.length; i2++) {
      let h = array[i2].toString(16);
      if (h.length < 2) {
        h = "0" + h;
      }
      str += h;
    }
    return str;
  }
};
const UINT32_MAX$1 = Math.pow(2, 32) - 1;
const push = [].push;
const RemuxerTrackIdConfig = {
  video: 1,
  audio: 2,
  id3: 3,
  text: 4
};
function bin2str(data) {
  return String.fromCharCode.apply(null, data);
}
function readUint16(buffer, offset) {
  const val = buffer[offset] << 8 | buffer[offset + 1];
  return val < 0 ? 65536 + val : val;
}
function readUint32(buffer, offset) {
  const val = readSint32(buffer, offset);
  return val < 0 ? 4294967296 + val : val;
}
function readUint64(buffer, offset) {
  let result = readUint32(buffer, offset);
  result *= Math.pow(2, 32);
  result += readUint32(buffer, offset + 4);
  return result;
}
function readSint32(buffer, offset) {
  return buffer[offset] << 24 | buffer[offset + 1] << 16 | buffer[offset + 2] << 8 | buffer[offset + 3];
}
function writeUint32(buffer, offset, value) {
  buffer[offset] = value >> 24;
  buffer[offset + 1] = value >> 16 & 255;
  buffer[offset + 2] = value >> 8 & 255;
  buffer[offset + 3] = value & 255;
}
function hasMoofData(data) {
  const end = data.byteLength;
  for (let i2 = 0; i2 < end; ) {
    const size = readUint32(data, i2);
    if (size > 8 && data[i2 + 4] === 109 && data[i2 + 5] === 111 && data[i2 + 6] === 111 && data[i2 + 7] === 102) {
      return true;
    }
    i2 = size > 1 ? i2 + size : end;
  }
  return false;
}
function findBox(data, path) {
  const results = [];
  if (!path.length) {
    return results;
  }
  const end = data.byteLength;
  for (let i2 = 0; i2 < end; ) {
    const size = readUint32(data, i2);
    const type = bin2str(data.subarray(i2 + 4, i2 + 8));
    const endbox = size > 1 ? i2 + size : end;
    if (type === path[0]) {
      if (path.length === 1) {
        results.push(data.subarray(i2 + 8, endbox));
      } else {
        const subresults = findBox(data.subarray(i2 + 8, endbox), path.slice(1));
        if (subresults.length) {
          push.apply(results, subresults);
        }
      }
    }
    i2 = endbox;
  }
  return results;
}
function parseSegmentIndex(sidx) {
  const references = [];
  const version = sidx[0];
  let index = 8;
  const timescale = readUint32(sidx, index);
  index += 4;
  let earliestPresentationTime = 0;
  let firstOffset = 0;
  if (version === 0) {
    earliestPresentationTime = readUint32(sidx, index);
    firstOffset = readUint32(sidx, index + 4);
    index += 8;
  } else {
    earliestPresentationTime = readUint64(sidx, index);
    firstOffset = readUint64(sidx, index + 8);
    index += 16;
  }
  index += 2;
  let startByte = sidx.length + firstOffset;
  const referencesCount = readUint16(sidx, index);
  index += 2;
  for (let i2 = 0; i2 < referencesCount; i2++) {
    let referenceIndex = index;
    const referenceInfo = readUint32(sidx, referenceIndex);
    referenceIndex += 4;
    const referenceSize = referenceInfo & 2147483647;
    const referenceType = (referenceInfo & 2147483648) >>> 31;
    if (referenceType === 1) {
      logger.warn("SIDX has hierarchical references (not supported)");
      return null;
    }
    const subsegmentDuration = readUint32(sidx, referenceIndex);
    referenceIndex += 4;
    references.push({
      referenceSize,
      subsegmentDuration,
      // unscaled
      info: {
        duration: subsegmentDuration / timescale,
        start: startByte,
        end: startByte + referenceSize - 1
      }
    });
    startByte += referenceSize;
    referenceIndex += 4;
    index = referenceIndex;
  }
  return {
    earliestPresentationTime,
    timescale,
    version,
    referencesCount,
    references
  };
}
function parseInitSegment(initSegment) {
  const result = [];
  const traks = findBox(initSegment, ["moov", "trak"]);
  for (let i2 = 0; i2 < traks.length; i2++) {
    const trak = traks[i2];
    const tkhd = findBox(trak, ["tkhd"])[0];
    if (tkhd) {
      let version = tkhd[0];
      const trackId = readUint32(tkhd, version === 0 ? 12 : 20);
      const mdhd = findBox(trak, ["mdia", "mdhd"])[0];
      if (mdhd) {
        version = mdhd[0];
        const timescale = readUint32(mdhd, version === 0 ? 12 : 20);
        const hdlr = findBox(trak, ["mdia", "hdlr"])[0];
        if (hdlr) {
          const hdlrType = bin2str(hdlr.subarray(8, 12));
          const type = {
            soun: ElementaryStreamTypes.AUDIO,
            vide: ElementaryStreamTypes.VIDEO
          }[hdlrType];
          if (type) {
            const stsd = findBox(trak, ["mdia", "minf", "stbl", "stsd"])[0];
            const stsdData = parseStsd(stsd);
            result[trackId] = {
              timescale,
              type
            };
            result[type] = _objectSpread2({
              timescale,
              id: trackId
            }, stsdData);
          }
        }
      }
    }
  }
  const trex = findBox(initSegment, ["moov", "mvex", "trex"]);
  trex.forEach((trex2) => {
    const trackId = readUint32(trex2, 4);
    const track = result[trackId];
    if (track) {
      track.default = {
        duration: readUint32(trex2, 12),
        flags: readUint32(trex2, 20)
      };
    }
  });
  return result;
}
function parseStsd(stsd) {
  const sampleEntries = stsd.subarray(8);
  const sampleEntriesEnd = sampleEntries.subarray(8 + 78);
  const fourCC = bin2str(sampleEntries.subarray(4, 8));
  let codec = fourCC;
  const encrypted = fourCC === "enca" || fourCC === "encv";
  if (encrypted) {
    const encBox = findBox(sampleEntries, [fourCC])[0];
    const encBoxChildren = encBox.subarray(fourCC === "enca" ? 28 : 78);
    const sinfs = findBox(encBoxChildren, ["sinf"]);
    sinfs.forEach((sinf) => {
      const schm = findBox(sinf, ["schm"])[0];
      if (schm) {
        const scheme = bin2str(schm.subarray(4, 8));
        if (scheme === "cbcs" || scheme === "cenc") {
          const frma = findBox(sinf, ["frma"])[0];
          if (frma) {
            codec = bin2str(frma);
          }
        }
      }
    });
  }
  switch (codec) {
    case "avc1":
    case "avc2":
    case "avc3":
    case "avc4": {
      const avcCBox = findBox(sampleEntriesEnd, ["avcC"])[0];
      codec += "." + toHex(avcCBox[1]) + toHex(avcCBox[2]) + toHex(avcCBox[3]);
      break;
    }
    case "mp4a": {
      const codecBox = findBox(sampleEntries, [fourCC])[0];
      const esdsBox = findBox(codecBox.subarray(28), ["esds"])[0];
      if (esdsBox && esdsBox.length > 12) {
        let i2 = 4;
        if (esdsBox[i2++] !== 3) {
          break;
        }
        i2 = skipBERInteger(esdsBox, i2);
        i2 += 2;
        const flags = esdsBox[i2++];
        if (flags & 128) {
          i2 += 2;
        }
        if (flags & 64) {
          i2 += esdsBox[i2++];
        }
        if (esdsBox[i2++] !== 4) {
          break;
        }
        i2 = skipBERInteger(esdsBox, i2);
        const objectType = esdsBox[i2++];
        if (objectType === 64) {
          codec += "." + toHex(objectType);
        } else {
          break;
        }
        i2 += 12;
        if (esdsBox[i2++] !== 5) {
          break;
        }
        i2 = skipBERInteger(esdsBox, i2);
        const firstByte = esdsBox[i2++];
        let audioObjectType = (firstByte & 248) >> 3;
        if (audioObjectType === 31) {
          audioObjectType += 1 + ((firstByte & 7) << 3) + ((esdsBox[i2] & 224) >> 5);
        }
        codec += "." + audioObjectType;
      }
      break;
    }
    case "hvc1":
    case "hev1": {
      const hvcCBox = findBox(sampleEntriesEnd, ["hvcC"])[0];
      const profileByte = hvcCBox[1];
      const profileSpace = ["", "A", "B", "C"][profileByte >> 6];
      const generalProfileIdc = profileByte & 31;
      const profileCompat = readUint32(hvcCBox, 2);
      const tierFlag = (profileByte & 32) >> 5 ? "H" : "L";
      const levelIDC = hvcCBox[12];
      const constraintIndicator = hvcCBox.subarray(6, 12);
      codec += "." + profileSpace + generalProfileIdc;
      codec += "." + profileCompat.toString(16).toUpperCase();
      codec += "." + tierFlag + levelIDC;
      let constraintString = "";
      for (let i2 = constraintIndicator.length; i2--; ) {
        const byte = constraintIndicator[i2];
        if (byte || constraintString) {
          const encodedByte = byte.toString(16).toUpperCase();
          constraintString = "." + encodedByte + constraintString;
        }
      }
      codec += constraintString;
      break;
    }
    case "dvh1":
    case "dvhe": {
      const dvcCBox = findBox(sampleEntriesEnd, ["dvcC"])[0];
      const profile = dvcCBox[2] >> 1 & 127;
      const level = dvcCBox[2] << 5 & 32 | dvcCBox[3] >> 3 & 31;
      codec += "." + addLeadingZero(profile) + "." + addLeadingZero(level);
      break;
    }
    case "vp09": {
      const vpcCBox = findBox(sampleEntriesEnd, ["vpcC"])[0];
      const profile = vpcCBox[4];
      const level = vpcCBox[5];
      const bitDepth = vpcCBox[6] >> 4 & 15;
      codec += "." + addLeadingZero(profile) + "." + addLeadingZero(level) + "." + addLeadingZero(bitDepth);
      break;
    }
    case "av01": {
      const av1CBox = findBox(sampleEntriesEnd, ["av1C"])[0];
      const profile = av1CBox[1] >>> 5;
      const level = av1CBox[1] & 31;
      const tierFlag = av1CBox[2] >>> 7 ? "H" : "M";
      const highBitDepth = (av1CBox[2] & 64) >> 6;
      const twelveBit = (av1CBox[2] & 32) >> 5;
      const bitDepth = profile === 2 && highBitDepth ? twelveBit ? 12 : 10 : highBitDepth ? 10 : 8;
      const monochrome = (av1CBox[2] & 16) >> 4;
      const chromaSubsamplingX = (av1CBox[2] & 8) >> 3;
      const chromaSubsamplingY = (av1CBox[2] & 4) >> 2;
      const chromaSamplePosition = av1CBox[2] & 3;
      const colorPrimaries = 1;
      const transferCharacteristics = 1;
      const matrixCoefficients = 1;
      const videoFullRangeFlag = 0;
      codec += "." + profile + "." + addLeadingZero(level) + tierFlag + "." + addLeadingZero(bitDepth) + "." + monochrome + "." + chromaSubsamplingX + chromaSubsamplingY + chromaSamplePosition + "." + addLeadingZero(colorPrimaries) + "." + addLeadingZero(transferCharacteristics) + "." + addLeadingZero(matrixCoefficients) + "." + videoFullRangeFlag;
      break;
    }
  }
  return {
    codec,
    encrypted
  };
}
function skipBERInteger(bytes, i2) {
  const limit = i2 + 5;
  while (bytes[i2++] & 128 && i2 < limit) {
  }
  return i2;
}
function toHex(x) {
  return ("0" + x.toString(16).toUpperCase()).slice(-2);
}
function addLeadingZero(num) {
  return (num < 10 ? "0" : "") + num;
}
function patchEncyptionData(initSegment, decryptdata) {
  if (!initSegment || !decryptdata) {
    return initSegment;
  }
  const keyId = decryptdata.keyId;
  if (keyId && decryptdata.isCommonEncryption) {
    const traks = findBox(initSegment, ["moov", "trak"]);
    traks.forEach((trak) => {
      const stsd = findBox(trak, ["mdia", "minf", "stbl", "stsd"])[0];
      const sampleEntries = stsd.subarray(8);
      let encBoxes = findBox(sampleEntries, ["enca"]);
      const isAudio = encBoxes.length > 0;
      if (!isAudio) {
        encBoxes = findBox(sampleEntries, ["encv"]);
      }
      encBoxes.forEach((enc) => {
        const encBoxChildren = isAudio ? enc.subarray(28) : enc.subarray(78);
        const sinfBoxes = findBox(encBoxChildren, ["sinf"]);
        sinfBoxes.forEach((sinf) => {
          const tenc = parseSinf(sinf);
          if (tenc) {
            const tencKeyId = tenc.subarray(8, 24);
            if (!tencKeyId.some((b) => b !== 0)) {
              logger.log(`[eme] Patching keyId in 'enc${isAudio ? "a" : "v"}>sinf>>tenc' box: ${Hex.hexDump(tencKeyId)} -> ${Hex.hexDump(keyId)}`);
              tenc.set(keyId, 8);
            }
          }
        });
      });
    });
  }
  return initSegment;
}
function parseSinf(sinf) {
  const schm = findBox(sinf, ["schm"])[0];
  if (schm) {
    const scheme = bin2str(schm.subarray(4, 8));
    if (scheme === "cbcs" || scheme === "cenc") {
      return findBox(sinf, ["schi", "tenc"])[0];
    }
  }
  logger.error(`[eme] missing 'schm' box`);
  return null;
}
function getStartDTS(initData, fmp4) {
  return findBox(fmp4, ["moof", "traf"]).reduce((result, traf) => {
    const tfdt = findBox(traf, ["tfdt"])[0];
    const version = tfdt[0];
    const start = findBox(traf, ["tfhd"]).reduce((result2, tfhd) => {
      const id = readUint32(tfhd, 4);
      const track = initData[id];
      if (track) {
        let baseTime = readUint32(tfdt, 4);
        if (version === 1) {
          if (baseTime === UINT32_MAX$1) {
            logger.warn(`[mp4-demuxer]: Ignoring assumed invalid signed 64-bit track fragment decode time`);
            return result2;
          }
          baseTime *= UINT32_MAX$1 + 1;
          baseTime += readUint32(tfdt, 8);
        }
        const scale = track.timescale || 9e4;
        const startTime = baseTime / scale;
        if (isFiniteNumber(startTime) && (result2 === null || startTime < result2)) {
          return startTime;
        }
      }
      return result2;
    }, null);
    if (start !== null && isFiniteNumber(start) && (result === null || start < result)) {
      return start;
    }
    return result;
  }, null);
}
function getDuration(data, initData) {
  let rawDuration = 0;
  let videoDuration = 0;
  let audioDuration = 0;
  const trafs = findBox(data, ["moof", "traf"]);
  for (let i2 = 0; i2 < trafs.length; i2++) {
    const traf = trafs[i2];
    const tfhd = findBox(traf, ["tfhd"])[0];
    const id = readUint32(tfhd, 4);
    const track = initData[id];
    if (!track) {
      continue;
    }
    const trackDefault = track.default;
    const tfhdFlags = readUint32(tfhd, 0) | (trackDefault == null ? void 0 : trackDefault.flags);
    let sampleDuration = trackDefault == null ? void 0 : trackDefault.duration;
    if (tfhdFlags & 8) {
      if (tfhdFlags & 2) {
        sampleDuration = readUint32(tfhd, 12);
      } else {
        sampleDuration = readUint32(tfhd, 8);
      }
    }
    const timescale = track.timescale || 9e4;
    const truns = findBox(traf, ["trun"]);
    for (let j = 0; j < truns.length; j++) {
      rawDuration = computeRawDurationFromSamples(truns[j]);
      if (!rawDuration && sampleDuration) {
        const sampleCount = readUint32(truns[j], 4);
        rawDuration = sampleDuration * sampleCount;
      }
      if (track.type === ElementaryStreamTypes.VIDEO) {
        videoDuration += rawDuration / timescale;
      } else if (track.type === ElementaryStreamTypes.AUDIO) {
        audioDuration += rawDuration / timescale;
      }
    }
  }
  if (videoDuration === 0 && audioDuration === 0) {
    let sidxMinStart = Infinity;
    let sidxMaxEnd = 0;
    let sidxDuration = 0;
    const sidxs = findBox(data, ["sidx"]);
    for (let i2 = 0; i2 < sidxs.length; i2++) {
      const sidx = parseSegmentIndex(sidxs[i2]);
      if (sidx != null && sidx.references) {
        sidxMinStart = Math.min(sidxMinStart, sidx.earliestPresentationTime / sidx.timescale);
        const subSegmentDuration = sidx.references.reduce((dur, ref) => dur + ref.info.duration || 0, 0);
        sidxMaxEnd = Math.max(sidxMaxEnd, subSegmentDuration + sidx.earliestPresentationTime / sidx.timescale);
        sidxDuration = sidxMaxEnd - sidxMinStart;
      }
    }
    if (sidxDuration && isFiniteNumber(sidxDuration)) {
      return sidxDuration;
    }
  }
  if (videoDuration) {
    return videoDuration;
  }
  return audioDuration;
}
function computeRawDurationFromSamples(trun) {
  const flags = readUint32(trun, 0);
  let offset = 8;
  if (flags & 1) {
    offset += 4;
  }
  if (flags & 4) {
    offset += 4;
  }
  let duration = 0;
  const sampleCount = readUint32(trun, 4);
  for (let i2 = 0; i2 < sampleCount; i2++) {
    if (flags & 256) {
      const sampleDuration = readUint32(trun, offset);
      duration += sampleDuration;
      offset += 4;
    }
    if (flags & 512) {
      offset += 4;
    }
    if (flags & 1024) {
      offset += 4;
    }
    if (flags & 2048) {
      offset += 4;
    }
  }
  return duration;
}
function offsetStartDTS(initData, fmp4, timeOffset) {
  findBox(fmp4, ["moof", "traf"]).forEach((traf) => {
    findBox(traf, ["tfhd"]).forEach((tfhd) => {
      const id = readUint32(tfhd, 4);
      const track = initData[id];
      if (!track) {
        return;
      }
      const timescale = track.timescale || 9e4;
      findBox(traf, ["tfdt"]).forEach((tfdt) => {
        const version = tfdt[0];
        const offset = timeOffset * timescale;
        if (offset) {
          let baseMediaDecodeTime = readUint32(tfdt, 4);
          if (version === 0) {
            baseMediaDecodeTime -= offset;
            baseMediaDecodeTime = Math.max(baseMediaDecodeTime, 0);
            writeUint32(tfdt, 4, baseMediaDecodeTime);
          } else {
            baseMediaDecodeTime *= Math.pow(2, 32);
            baseMediaDecodeTime += readUint32(tfdt, 8);
            baseMediaDecodeTime -= offset;
            baseMediaDecodeTime = Math.max(baseMediaDecodeTime, 0);
            const upper = Math.floor(baseMediaDecodeTime / (UINT32_MAX$1 + 1));
            const lower = Math.floor(baseMediaDecodeTime % (UINT32_MAX$1 + 1));
            writeUint32(tfdt, 4, upper);
            writeUint32(tfdt, 8, lower);
          }
        }
      });
    });
  });
}
function segmentValidRange(data) {
  const segmentedRange = {
    valid: null,
    remainder: null
  };
  const moofs = findBox(data, ["moof"]);
  if (moofs.length < 2) {
    segmentedRange.remainder = data;
    return segmentedRange;
  }
  const last = moofs[moofs.length - 1];
  segmentedRange.valid = sliceUint8(data, 0, last.byteOffset - 8);
  segmentedRange.remainder = sliceUint8(data, last.byteOffset - 8);
  return segmentedRange;
}
function appendUint8Array(data1, data2) {
  const temp = new Uint8Array(data1.length + data2.length);
  temp.set(data1);
  temp.set(data2, data1.length);
  return temp;
}
function parseSamples(timeOffset, track) {
  const seiSamples = [];
  const videoData = track.samples;
  const timescale = track.timescale;
  const trackId = track.id;
  let isHEVCFlavor = false;
  const moofs = findBox(videoData, ["moof"]);
  moofs.map((moof) => {
    const moofOffset = moof.byteOffset - 8;
    const trafs = findBox(moof, ["traf"]);
    trafs.map((traf) => {
      const baseTime = findBox(traf, ["tfdt"]).map((tfdt) => {
        const version = tfdt[0];
        let result = readUint32(tfdt, 4);
        if (version === 1) {
          result *= Math.pow(2, 32);
          result += readUint32(tfdt, 8);
        }
        return result / timescale;
      })[0];
      if (baseTime !== void 0) {
        timeOffset = baseTime;
      }
      return findBox(traf, ["tfhd"]).map((tfhd) => {
        const id = readUint32(tfhd, 4);
        const tfhdFlags = readUint32(tfhd, 0) & 16777215;
        const baseDataOffsetPresent = (tfhdFlags & 1) !== 0;
        const sampleDescriptionIndexPresent = (tfhdFlags & 2) !== 0;
        const defaultSampleDurationPresent = (tfhdFlags & 8) !== 0;
        let defaultSampleDuration = 0;
        const defaultSampleSizePresent = (tfhdFlags & 16) !== 0;
        let defaultSampleSize = 0;
        const defaultSampleFlagsPresent = (tfhdFlags & 32) !== 0;
        let tfhdOffset = 8;
        if (id === trackId) {
          if (baseDataOffsetPresent) {
            tfhdOffset += 8;
          }
          if (sampleDescriptionIndexPresent) {
            tfhdOffset += 4;
          }
          if (defaultSampleDurationPresent) {
            defaultSampleDuration = readUint32(tfhd, tfhdOffset);
            tfhdOffset += 4;
          }
          if (defaultSampleSizePresent) {
            defaultSampleSize = readUint32(tfhd, tfhdOffset);
            tfhdOffset += 4;
          }
          if (defaultSampleFlagsPresent) {
            tfhdOffset += 4;
          }
          if (track.type === "video") {
            isHEVCFlavor = isHEVC(track.codec);
          }
          findBox(traf, ["trun"]).map((trun) => {
            const version = trun[0];
            const flags = readUint32(trun, 0) & 16777215;
            const dataOffsetPresent = (flags & 1) !== 0;
            let dataOffset = 0;
            const firstSampleFlagsPresent = (flags & 4) !== 0;
            const sampleDurationPresent = (flags & 256) !== 0;
            let sampleDuration = 0;
            const sampleSizePresent = (flags & 512) !== 0;
            let sampleSize = 0;
            const sampleFlagsPresent = (flags & 1024) !== 0;
            const sampleCompositionOffsetsPresent = (flags & 2048) !== 0;
            let compositionOffset = 0;
            const sampleCount = readUint32(trun, 4);
            let trunOffset = 8;
            if (dataOffsetPresent) {
              dataOffset = readUint32(trun, trunOffset);
              trunOffset += 4;
            }
            if (firstSampleFlagsPresent) {
              trunOffset += 4;
            }
            let sampleOffset = dataOffset + moofOffset;
            for (let ix = 0; ix < sampleCount; ix++) {
              if (sampleDurationPresent) {
                sampleDuration = readUint32(trun, trunOffset);
                trunOffset += 4;
              } else {
                sampleDuration = defaultSampleDuration;
              }
              if (sampleSizePresent) {
                sampleSize = readUint32(trun, trunOffset);
                trunOffset += 4;
              } else {
                sampleSize = defaultSampleSize;
              }
              if (sampleFlagsPresent) {
                trunOffset += 4;
              }
              if (sampleCompositionOffsetsPresent) {
                if (version === 0) {
                  compositionOffset = readUint32(trun, trunOffset);
                } else {
                  compositionOffset = readSint32(trun, trunOffset);
                }
                trunOffset += 4;
              }
              if (track.type === ElementaryStreamTypes.VIDEO) {
                let naluTotalSize = 0;
                while (naluTotalSize < sampleSize) {
                  const naluSize = readUint32(videoData, sampleOffset);
                  sampleOffset += 4;
                  if (isSEIMessage(isHEVCFlavor, videoData[sampleOffset])) {
                    const data = videoData.subarray(sampleOffset, sampleOffset + naluSize);
                    parseSEIMessageFromNALu(data, isHEVCFlavor ? 2 : 1, timeOffset + compositionOffset / timescale, seiSamples);
                  }
                  sampleOffset += naluSize;
                  naluTotalSize += naluSize + 4;
                }
              }
              timeOffset += sampleDuration / timescale;
            }
          });
        }
      });
    });
  });
  return seiSamples;
}
function isHEVC(codec) {
  if (!codec) {
    return false;
  }
  const delimit = codec.indexOf(".");
  const baseCodec = delimit < 0 ? codec : codec.substring(0, delimit);
  return baseCodec === "hvc1" || baseCodec === "hev1" || // Dolby Vision
  baseCodec === "dvh1" || baseCodec === "dvhe";
}
function isSEIMessage(isHEVCFlavor, naluHeader) {
  if (isHEVCFlavor) {
    const naluType = naluHeader >> 1 & 63;
    return naluType === 39 || naluType === 40;
  } else {
    const naluType = naluHeader & 31;
    return naluType === 6;
  }
}
function parseSEIMessageFromNALu(unescapedData, headerSize, pts, samples) {
  const data = discardEPB(unescapedData);
  let seiPtr = 0;
  seiPtr += headerSize;
  let payloadType = 0;
  let payloadSize = 0;
  let b = 0;
  while (seiPtr < data.length) {
    payloadType = 0;
    do {
      if (seiPtr >= data.length) {
        break;
      }
      b = data[seiPtr++];
      payloadType += b;
    } while (b === 255);
    payloadSize = 0;
    do {
      if (seiPtr >= data.length) {
        break;
      }
      b = data[seiPtr++];
      payloadSize += b;
    } while (b === 255);
    const leftOver = data.length - seiPtr;
    let payPtr = seiPtr;
    if (payloadSize < leftOver) {
      seiPtr += payloadSize;
    } else if (payloadSize > leftOver) {
      logger.error(`Malformed SEI payload. ${payloadSize} is too small, only ${leftOver} bytes left to parse.`);
      break;
    }
    if (payloadType === 4) {
      const countryCode = data[payPtr++];
      if (countryCode === 181) {
        const providerCode = readUint16(data, payPtr);
        payPtr += 2;
        if (providerCode === 49) {
          const userStructure = readUint32(data, payPtr);
          payPtr += 4;
          if (userStructure === 1195456820) {
            const userDataType = data[payPtr++];
            if (userDataType === 3) {
              const firstByte = data[payPtr++];
              const totalCCs = 31 & firstByte;
              const enabled = 64 & firstByte;
              const totalBytes = enabled ? 2 + totalCCs * 3 : 0;
              const byteArray = new Uint8Array(totalBytes);
              if (enabled) {
                byteArray[0] = firstByte;
                for (let i2 = 1; i2 < totalBytes; i2++) {
                  byteArray[i2] = data[payPtr++];
                }
              }
              samples.push({
                type: userDataType,
                payloadType,
                pts,
                bytes: byteArray
              });
            }
          }
        }
      }
    } else if (payloadType === 5) {
      if (payloadSize > 16) {
        const uuidStrArray = [];
        for (let i2 = 0; i2 < 16; i2++) {
          const _b = data[payPtr++].toString(16);
          uuidStrArray.push(_b.length == 1 ? "0" + _b : _b);
          if (i2 === 3 || i2 === 5 || i2 === 7 || i2 === 9) {
            uuidStrArray.push("-");
          }
        }
        const length = payloadSize - 16;
        const userDataBytes = new Uint8Array(length);
        for (let i2 = 0; i2 < length; i2++) {
          userDataBytes[i2] = data[payPtr++];
        }
        samples.push({
          payloadType,
          pts,
          uuid: uuidStrArray.join(""),
          userData: utf8ArrayToStr(userDataBytes),
          userDataBytes
        });
      }
    }
  }
}
function discardEPB(data) {
  const length = data.byteLength;
  const EPBPositions = [];
  let i2 = 1;
  while (i2 < length - 2) {
    if (data[i2] === 0 && data[i2 + 1] === 0 && data[i2 + 2] === 3) {
      EPBPositions.push(i2 + 2);
      i2 += 2;
    } else {
      i2++;
    }
  }
  if (EPBPositions.length === 0) {
    return data;
  }
  const newLength = length - EPBPositions.length;
  const newData = new Uint8Array(newLength);
  let sourceIndex = 0;
  for (i2 = 0; i2 < newLength; sourceIndex++, i2++) {
    if (sourceIndex === EPBPositions[0]) {
      sourceIndex++;
      EPBPositions.shift();
    }
    newData[i2] = data[sourceIndex];
  }
  return newData;
}
function parseEmsg(data) {
  const version = data[0];
  let schemeIdUri = "";
  let value = "";
  let timeScale = 0;
  let presentationTimeDelta = 0;
  let presentationTime = 0;
  let eventDuration = 0;
  let id = 0;
  let offset = 0;
  if (version === 0) {
    while (bin2str(data.subarray(offset, offset + 1)) !== "\0") {
      schemeIdUri += bin2str(data.subarray(offset, offset + 1));
      offset += 1;
    }
    schemeIdUri += bin2str(data.subarray(offset, offset + 1));
    offset += 1;
    while (bin2str(data.subarray(offset, offset + 1)) !== "\0") {
      value += bin2str(data.subarray(offset, offset + 1));
      offset += 1;
    }
    value += bin2str(data.subarray(offset, offset + 1));
    offset += 1;
    timeScale = readUint32(data, 12);
    presentationTimeDelta = readUint32(data, 16);
    eventDuration = readUint32(data, 20);
    id = readUint32(data, 24);
    offset = 28;
  } else if (version === 1) {
    offset += 4;
    timeScale = readUint32(data, offset);
    offset += 4;
    const leftPresentationTime = readUint32(data, offset);
    offset += 4;
    const rightPresentationTime = readUint32(data, offset);
    offset += 4;
    presentationTime = 2 ** 32 * leftPresentationTime + rightPresentationTime;
    if (!isSafeInteger(presentationTime)) {
      presentationTime = Number.MAX_SAFE_INTEGER;
      logger.warn("Presentation time exceeds safe integer limit and wrapped to max safe integer in parsing emsg box");
    }
    eventDuration = readUint32(data, offset);
    offset += 4;
    id = readUint32(data, offset);
    offset += 4;
    while (bin2str(data.subarray(offset, offset + 1)) !== "\0") {
      schemeIdUri += bin2str(data.subarray(offset, offset + 1));
      offset += 1;
    }
    schemeIdUri += bin2str(data.subarray(offset, offset + 1));
    offset += 1;
    while (bin2str(data.subarray(offset, offset + 1)) !== "\0") {
      value += bin2str(data.subarray(offset, offset + 1));
      offset += 1;
    }
    value += bin2str(data.subarray(offset, offset + 1));
    offset += 1;
  }
  const payload = data.subarray(offset, data.byteLength);
  return {
    schemeIdUri,
    value,
    timeScale,
    presentationTime,
    presentationTimeDelta,
    eventDuration,
    id,
    payload
  };
}
function mp4Box(type, ...payload) {
  const len = payload.length;
  let size = 8;
  let i2 = len;
  while (i2--) {
    size += payload[i2].byteLength;
  }
  const result = new Uint8Array(size);
  result[0] = size >> 24 & 255;
  result[1] = size >> 16 & 255;
  result[2] = size >> 8 & 255;
  result[3] = size & 255;
  result.set(type, 4);
  for (i2 = 0, size = 8; i2 < len; i2++) {
    result.set(payload[i2], size);
    size += payload[i2].byteLength;
  }
  return result;
}
function mp4pssh(systemId, keyids, data) {
  if (systemId.byteLength !== 16) {
    throw new RangeError("Invalid system id");
  }
  let version;
  let kids;
  {
    version = 0;
    kids = new Uint8Array();
  }
  let kidCount;
  if (version > 0) {
    kidCount = new Uint8Array(4);
    if (keyids.length > 0) {
      new DataView(kidCount.buffer).setUint32(0, keyids.length, false);
    }
  } else {
    kidCount = new Uint8Array();
  }
  const dataSize = new Uint8Array(4);
  if (data && data.byteLength > 0) {
    new DataView(dataSize.buffer).setUint32(0, data.byteLength, false);
  }
  return mp4Box(
    [112, 115, 115, 104],
    new Uint8Array([
      version,
      0,
      0,
      0
      // Flags
    ]),
    systemId,
    // 16 bytes
    kidCount,
    kids,
    dataSize,
    data || new Uint8Array()
  );
}
function parsePssh(initData) {
  if (!(initData instanceof ArrayBuffer) || initData.byteLength < 32) {
    return null;
  }
  const result = {
    version: 0,
    systemId: "",
    kids: null,
    data: null
  };
  const view = new DataView(initData);
  const boxSize = view.getUint32(0);
  if (initData.byteLength !== boxSize && boxSize > 44) {
    return null;
  }
  const type = view.getUint32(4);
  if (type !== 1886614376) {
    return null;
  }
  result.version = view.getUint32(8) >>> 24;
  if (result.version > 1) {
    return null;
  }
  result.systemId = Hex.hexDump(new Uint8Array(initData, 12, 16));
  const dataSizeOrKidCount = view.getUint32(28);
  if (result.version === 0) {
    if (boxSize - 32 < dataSizeOrKidCount) {
      return null;
    }
    result.data = new Uint8Array(initData, 32, dataSizeOrKidCount);
  } else if (result.version === 1) {
    result.kids = [];
    for (let i2 = 0; i2 < dataSizeOrKidCount; i2++) {
      result.kids.push(new Uint8Array(initData, 32 + i2 * 16, 16));
    }
  }
  return result;
}
let keyUriToKeyIdMap = {};
class LevelKey {
  static clearKeyUriToKeyIdMap() {
    keyUriToKeyIdMap = {};
  }
  constructor(method, uri, format2, formatversions = [1], iv = null) {
    this.uri = void 0;
    this.method = void 0;
    this.keyFormat = void 0;
    this.keyFormatVersions = void 0;
    this.encrypted = void 0;
    this.isCommonEncryption = void 0;
    this.iv = null;
    this.key = null;
    this.keyId = null;
    this.pssh = null;
    this.method = method;
    this.uri = uri;
    this.keyFormat = format2;
    this.keyFormatVersions = formatversions;
    this.iv = iv;
    this.encrypted = method ? method !== "NONE" : false;
    this.isCommonEncryption = this.encrypted && method !== "AES-128";
  }
  isSupported() {
    if (this.method) {
      if (this.method === "AES-128" || this.method === "NONE") {
        return true;
      }
      if (this.keyFormat === "identity") {
        return this.method === "SAMPLE-AES";
      } else {
        switch (this.keyFormat) {
          case KeySystemFormats.FAIRPLAY:
          case KeySystemFormats.WIDEVINE:
          case KeySystemFormats.PLAYREADY:
          case KeySystemFormats.CLEARKEY:
            return ["ISO-23001-7", "SAMPLE-AES", "SAMPLE-AES-CENC", "SAMPLE-AES-CTR"].indexOf(this.method) !== -1;
        }
      }
    }
    return false;
  }
  getDecryptData(sn) {
    if (!this.encrypted || !this.uri) {
      return null;
    }
    if (this.method === "AES-128" && this.uri && !this.iv) {
      if (typeof sn !== "number") {
        if (this.method === "AES-128" && !this.iv) {
          logger.warn(`missing IV for initialization segment with method="${this.method}" - compliance issue`);
        }
        sn = 0;
      }
      const iv = createInitializationVector(sn);
      const decryptdata = new LevelKey(this.method, this.uri, "identity", this.keyFormatVersions, iv);
      return decryptdata;
    }
    const keyBytes = convertDataUriToArrayBytes(this.uri);
    if (keyBytes) {
      switch (this.keyFormat) {
        case KeySystemFormats.WIDEVINE:
          this.pssh = keyBytes;
          if (keyBytes.length >= 22) {
            this.keyId = keyBytes.subarray(keyBytes.length - 22, keyBytes.length - 6);
          }
          break;
        case KeySystemFormats.PLAYREADY: {
          const PlayReadyKeySystemUUID = new Uint8Array([154, 4, 240, 121, 152, 64, 66, 134, 171, 146, 230, 91, 224, 136, 95, 149]);
          this.pssh = mp4pssh(PlayReadyKeySystemUUID, null, keyBytes);
          const keyBytesUtf16 = new Uint16Array(keyBytes.buffer, keyBytes.byteOffset, keyBytes.byteLength / 2);
          const keyByteStr = String.fromCharCode.apply(null, Array.from(keyBytesUtf16));
          const xmlKeyBytes = keyByteStr.substring(keyByteStr.indexOf("<"), keyByteStr.length);
          const parser = new DOMParser();
          const xmlDoc = parser.parseFromString(xmlKeyBytes, "text/xml");
          const keyData = xmlDoc.getElementsByTagName("KID")[0];
          if (keyData) {
            const keyId = keyData.childNodes[0] ? keyData.childNodes[0].nodeValue : keyData.getAttribute("VALUE");
            if (keyId) {
              const keyIdArray = base64Decode(keyId).subarray(0, 16);
              changeEndianness(keyIdArray);
              this.keyId = keyIdArray;
            }
          }
          break;
        }
        default: {
          let keydata = keyBytes.subarray(0, 16);
          if (keydata.length !== 16) {
            const padded = new Uint8Array(16);
            padded.set(keydata, 16 - keydata.length);
            keydata = padded;
          }
          this.keyId = keydata;
          break;
        }
      }
    }
    if (!this.keyId || this.keyId.byteLength !== 16) {
      let keyId = keyUriToKeyIdMap[this.uri];
      if (!keyId) {
        const val = Object.keys(keyUriToKeyIdMap).length % Number.MAX_SAFE_INTEGER;
        keyId = new Uint8Array(16);
        const dv = new DataView(keyId.buffer, 12, 4);
        dv.setUint32(0, val);
        keyUriToKeyIdMap[this.uri] = keyId;
      }
      this.keyId = keyId;
    }
    return this;
  }
}
function createInitializationVector(segmentNumber) {
  const uint8View = new Uint8Array(16);
  for (let i2 = 12; i2 < 16; i2++) {
    uint8View[i2] = segmentNumber >> 8 * (15 - i2) & 255;
  }
  return uint8View;
}
function getMediaSource(preferManagedMediaSource = true) {
  if (typeof self === "undefined") return void 0;
  const mms = (preferManagedMediaSource || !self.MediaSource) && self.ManagedMediaSource;
  return mms || self.MediaSource || self.WebKitMediaSource;
}
function isManagedMediaSource(source) {
  return typeof self !== "undefined" && source === self.ManagedMediaSource;
}
const sampleEntryCodesISO = {
  audio: {
    a3ds: 1,
    "ac-3": 0.95,
    "ac-4": 1,
    alac: 0.9,
    alaw: 1,
    dra1: 1,
    "dts+": 1,
    "dts-": 1,
    dtsc: 1,
    dtse: 1,
    dtsh: 1,
    "ec-3": 0.9,
    enca: 1,
    fLaC: 0.9,
    // MP4-RA listed codec entry for FLAC
    flac: 0.9,
    // legacy browser codec name for FLAC
    FLAC: 0.9,
    // some manifests may list "FLAC" with Apple's tools
    g719: 1,
    g726: 1,
    m4ae: 1,
    mha1: 1,
    mha2: 1,
    mhm1: 1,
    mhm2: 1,
    mlpa: 1,
    mp4a: 1,
    "raw ": 1,
    Opus: 1,
    opus: 1,
    // browsers expect this to be lowercase despite MP4RA says 'Opus'
    samr: 1,
    sawb: 1,
    sawp: 1,
    sevc: 1,
    sqcp: 1,
    ssmv: 1,
    twos: 1,
    ulaw: 1
  },
  video: {
    avc1: 1,
    avc2: 1,
    avc3: 1,
    avc4: 1,
    avcp: 1,
    av01: 0.8,
    drac: 1,
    dva1: 1,
    dvav: 1,
    dvh1: 0.7,
    dvhe: 0.7,
    encv: 1,
    hev1: 0.75,
    hvc1: 0.75,
    mjp2: 1,
    mp4v: 1,
    mvc1: 1,
    mvc2: 1,
    mvc3: 1,
    mvc4: 1,
    resv: 1,
    rv60: 1,
    s263: 1,
    svc1: 1,
    svc2: 1,
    "vc-1": 1,
    vp08: 1,
    vp09: 0.9
  }
};
function isCodecMediaSourceSupported(codec, type, preferManagedMediaSource = true) {
  var _MediaSource$isTypeSu;
  const MediaSource = getMediaSource(preferManagedMediaSource);
  return (_MediaSource$isTypeSu = MediaSource == null ? void 0 : MediaSource.isTypeSupported(mimeTypeForCodec(codec, type))) != null ? _MediaSource$isTypeSu : false;
}
function mimeTypeForCodec(codec, type) {
  return `${type}/mp4;codecs="${codec}"`;
}
function codecsSetSelectionPreferenceValue(codecSet) {
  return codecSet.split(",").reduce((num, fourCC) => {
    const preferenceValue = sampleEntryCodesISO.video[fourCC];
    if (preferenceValue) {
      return (preferenceValue * 2 + num) / (num ? 3 : 2);
    }
    return (sampleEntryCodesISO.audio[fourCC] + num) / (num ? 2 : 1);
  }, 0);
}
const CODEC_COMPATIBLE_NAMES = {};
function getCodecCompatibleNameLower(lowerCaseCodec, preferManagedMediaSource = true) {
  if (CODEC_COMPATIBLE_NAMES[lowerCaseCodec]) {
    return CODEC_COMPATIBLE_NAMES[lowerCaseCodec];
  }
  const codecsToCheck = {
    flac: ["flac", "fLaC", "FLAC"],
    opus: ["opus", "Opus"]
  }[lowerCaseCodec];
  for (let i2 = 0; i2 < codecsToCheck.length; i2++) {
    if (isCodecMediaSourceSupported(codecsToCheck[i2], "audio", preferManagedMediaSource)) {
      CODEC_COMPATIBLE_NAMES[lowerCaseCodec] = codecsToCheck[i2];
      return codecsToCheck[i2];
    }
  }
  return lowerCaseCodec;
}
const AUDIO_CODEC_REGEXP = /flac|opus/i;
function getCodecCompatibleName(codec, preferManagedMediaSource = true) {
  return codec.replace(AUDIO_CODEC_REGEXP, (m) => getCodecCompatibleNameLower(m.toLowerCase(), preferManagedMediaSource));
}
function pickMostCompleteCodecName(parsedCodec, levelCodec) {
  if (parsedCodec && parsedCodec !== "mp4a") {
    return parsedCodec;
  }
  return levelCodec ? levelCodec.split(",")[0] : levelCodec;
}
var PlaylistContextType = {
  AUDIO_TRACK: "audioTrack",
  SUBTITLE_TRACK: "subtitleTrack"
};
var PlaylistLevelType = {
  MAIN: "main",
  AUDIO: "audio",
  SUBTITLE: "subtitle"
};
function sendAddTrackEvent(track, videoEl) {
  let event;
  try {
    event = new Event("addtrack");
  } catch (err) {
    event = document.createEvent("Event");
    event.initEvent("addtrack", false, false);
  }
  event.track = track;
  videoEl.dispatchEvent(event);
}
function addCueToTrack(track, cue) {
  const mode = track.mode;
  if (mode === "disabled") {
    track.mode = "hidden";
  }
  if (track.cues && !track.cues.getCueById(cue.id)) {
    try {
      track.addCue(cue);
      if (!track.cues.getCueById(cue.id)) {
        throw new Error(`addCue is failed for: ${cue}`);
      }
    } catch (err) {
      logger.debug(`[texttrack-utils]: ${err}`);
      try {
        const textTrackCue = new self.TextTrackCue(cue.startTime, cue.endTime, cue.text);
        textTrackCue.id = cue.id;
        track.addCue(textTrackCue);
      } catch (err2) {
        logger.debug(`[texttrack-utils]: Legacy TextTrackCue fallback failed: ${err2}`);
      }
    }
  }
  if (mode === "disabled") {
    track.mode = mode;
  }
}
function clearCurrentCues(track) {
  const mode = track.mode;
  if (mode === "disabled") {
    track.mode = "hidden";
  }
  if (track.cues) {
    for (let i2 = track.cues.length; i2--; ) {
      track.removeCue(track.cues[i2]);
    }
  }
  if (mode === "disabled") {
    track.mode = mode;
  }
}
function removeCuesInRange(track, start, end, predicate) {
  const mode = track.mode;
  if (mode === "disabled") {
    track.mode = "hidden";
  }
  if (track.cues && track.cues.length > 0) {
    const cues = getCuesInRange(track.cues, start, end);
    for (let i2 = 0; i2 < cues.length; i2++) {
      {
        track.removeCue(cues[i2]);
      }
    }
  }
  if (mode === "disabled") {
    track.mode = mode;
  }
}
function getFirstCueIndexAfterTime(cues, time) {
  if (time < cues[0].startTime) {
    return 0;
  }
  const len = cues.length - 1;
  if (time > cues[len].endTime) {
    return -1;
  }
  let left = 0;
  let right = len;
  while (left <= right) {
    const mid = Math.floor((right + left) / 2);
    if (time < cues[mid].startTime) {
      right = mid - 1;
    } else if (time > cues[mid].startTime && left < len) {
      left = mid + 1;
    } else {
      return mid;
    }
  }
  return cues[left].startTime - time < time - cues[right].startTime ? left : right;
}
function getCuesInRange(cues, start, end) {
  const cuesFound = [];
  const firstCueInRange = getFirstCueIndexAfterTime(cues, start);
  if (firstCueInRange > -1) {
    for (let i2 = firstCueInRange, len = cues.length; i2 < len; i2++) {
      const cue = cues[i2];
      if (cue.startTime >= start && cue.endTime <= end) {
        cuesFound.push(cue);
      } else if (cue.startTime > end) {
        return cuesFound;
      }
    }
  }
  return cuesFound;
}
function filterSubtitleTracks(textTrackList) {
  const tracks = [];
  for (let i2 = 0; i2 < textTrackList.length; i2++) {
    const track = textTrackList[i2];
    if ((track.kind === "subtitles" || track.kind === "captions") && track.label) {
      tracks.push(textTrackList[i2]);
    }
  }
  return tracks;
}
var MetadataSchema = {
  audioId3: "org.id3",
  emsg: "https://aomedia.org/emsg/ID3"
};
function getCueClass() {
  if (typeof self === "undefined") return void 0;
  return self.VTTCue || self.TextTrackCue;
}
(() => {
  const Cue = getCueClass();
  try {
    Cue && new Cue(0, Number.POSITIVE_INFINITY, "");
  } catch (e2) {
    return Number.MAX_VALUE;
  }
  return Number.POSITIVE_INFINITY;
})();
const HdcpLevels = ["NONE", "TYPE-0", "TYPE-1", null];
const VideoRangeValues = ["SDR", "PQ", "HLG"];
var HlsSkip = {
  No: "",
  Yes: "YES",
  v2: "v2"
};
function getSkipValue(details) {
  const {
    canSkipUntil,
    canSkipDateRanges,
    age
  } = details;
  const playlistRecentEnough = age < canSkipUntil / 2;
  if (canSkipUntil && playlistRecentEnough) {
    if (canSkipDateRanges) {
      return HlsSkip.v2;
    }
    return HlsSkip.Yes;
  }
  return HlsSkip.No;
}
class HlsUrlParameters {
  constructor(msn, part, skip) {
    this.msn = void 0;
    this.part = void 0;
    this.skip = void 0;
    this.msn = msn;
    this.part = part;
    this.skip = skip;
  }
  addDirectives(uri) {
    const url = new self.URL(uri);
    if (this.msn !== void 0) {
      url.searchParams.set("_HLS_msn", this.msn.toString());
    }
    if (this.part !== void 0) {
      url.searchParams.set("_HLS_part", this.part.toString());
    }
    if (this.skip) {
      url.searchParams.set("_HLS_skip", this.skip);
    }
    return url.href;
  }
}
class Level {
  constructor(data) {
    this._attrs = void 0;
    this.audioCodec = void 0;
    this.bitrate = void 0;
    this.codecSet = void 0;
    this.url = void 0;
    this.frameRate = void 0;
    this.height = void 0;
    this.id = void 0;
    this.name = void 0;
    this.videoCodec = void 0;
    this.width = void 0;
    this.details = void 0;
    this.fragmentError = 0;
    this.loadError = 0;
    this.loaded = void 0;
    this.realBitrate = 0;
    this.supportedPromise = void 0;
    this.supportedResult = void 0;
    this._avgBitrate = 0;
    this._audioGroups = void 0;
    this._subtitleGroups = void 0;
    this._urlId = 0;
    this.url = [data.url];
    this._attrs = [data.attrs];
    this.bitrate = data.bitrate;
    if (data.details) {
      this.details = data.details;
    }
    this.id = data.id || 0;
    this.name = data.name;
    this.width = data.width || 0;
    this.height = data.height || 0;
    this.frameRate = data.attrs.optionalFloat("FRAME-RATE", 0);
    this._avgBitrate = data.attrs.decimalInteger("AVERAGE-BANDWIDTH");
    this.audioCodec = data.audioCodec;
    this.videoCodec = data.videoCodec;
    this.codecSet = [data.videoCodec, data.audioCodec].filter((c) => !!c).map((s2) => s2.substring(0, 4)).join(",");
    this.addGroupId("audio", data.attrs.AUDIO);
    this.addGroupId("text", data.attrs.SUBTITLES);
  }
  get maxBitrate() {
    return Math.max(this.realBitrate, this.bitrate);
  }
  get averageBitrate() {
    return this._avgBitrate || this.realBitrate || this.bitrate;
  }
  get attrs() {
    return this._attrs[0];
  }
  get codecs() {
    return this.attrs.CODECS || "";
  }
  get pathwayId() {
    return this.attrs["PATHWAY-ID"] || ".";
  }
  get videoRange() {
    return this.attrs["VIDEO-RANGE"] || "SDR";
  }
  get score() {
    return this.attrs.optionalFloat("SCORE", 0);
  }
  get uri() {
    return this.url[0] || "";
  }
  hasAudioGroup(groupId) {
    return hasGroup(this._audioGroups, groupId);
  }
  hasSubtitleGroup(groupId) {
    return hasGroup(this._subtitleGroups, groupId);
  }
  get audioGroups() {
    return this._audioGroups;
  }
  get subtitleGroups() {
    return this._subtitleGroups;
  }
  addGroupId(type, groupId) {
    if (!groupId) {
      return;
    }
    if (type === "audio") {
      let audioGroups = this._audioGroups;
      if (!audioGroups) {
        audioGroups = this._audioGroups = [];
      }
      if (audioGroups.indexOf(groupId) === -1) {
        audioGroups.push(groupId);
      }
    } else if (type === "text") {
      let subtitleGroups = this._subtitleGroups;
      if (!subtitleGroups) {
        subtitleGroups = this._subtitleGroups = [];
      }
      if (subtitleGroups.indexOf(groupId) === -1) {
        subtitleGroups.push(groupId);
      }
    }
  }
  // Deprecated methods (retained for backwards compatibility)
  get urlId() {
    return 0;
  }
  set urlId(value) {
  }
  get audioGroupIds() {
    return this.audioGroups ? [this.audioGroupId] : void 0;
  }
  get textGroupIds() {
    return this.subtitleGroups ? [this.textGroupId] : void 0;
  }
  get audioGroupId() {
    var _this$audioGroups;
    return (_this$audioGroups = this.audioGroups) == null ? void 0 : _this$audioGroups[0];
  }
  get textGroupId() {
    var _this$subtitleGroups;
    return (_this$subtitleGroups = this.subtitleGroups) == null ? void 0 : _this$subtitleGroups[0];
  }
  addFallback() {
  }
}
function hasGroup(groups, groupId) {
  if (!groupId || !groups) {
    return false;
  }
  return groups.indexOf(groupId) !== -1;
}
function updateFromToPTS(fragFrom, fragTo) {
  const fragToPTS = fragTo.startPTS;
  if (isFiniteNumber(fragToPTS)) {
    let duration = 0;
    let frag;
    if (fragTo.sn > fragFrom.sn) {
      duration = fragToPTS - fragFrom.start;
      frag = fragFrom;
    } else {
      duration = fragFrom.start - fragToPTS;
      frag = fragTo;
    }
    if (frag.duration !== duration) {
      frag.duration = duration;
    }
  } else if (fragTo.sn > fragFrom.sn) {
    const contiguous = fragFrom.cc === fragTo.cc;
    if (contiguous && fragFrom.minEndPTS) {
      fragTo.start = fragFrom.start + (fragFrom.minEndPTS - fragFrom.start);
    } else {
      fragTo.start = fragFrom.start + fragFrom.duration;
    }
  } else {
    fragTo.start = Math.max(fragFrom.start - fragTo.duration, 0);
  }
}
function updateFragPTSDTS(details, frag, startPTS, endPTS, startDTS, endDTS) {
  const parsedMediaDuration = endPTS - startPTS;
  if (parsedMediaDuration <= 0) {
    logger.warn("Fragment should have a positive duration", frag);
    endPTS = startPTS + frag.duration;
    endDTS = startDTS + frag.duration;
  }
  let maxStartPTS = startPTS;
  let minEndPTS = endPTS;
  const fragStartPts = frag.startPTS;
  const fragEndPts = frag.endPTS;
  if (isFiniteNumber(fragStartPts)) {
    const deltaPTS = Math.abs(fragStartPts - startPTS);
    if (!isFiniteNumber(frag.deltaPTS)) {
      frag.deltaPTS = deltaPTS;
    } else {
      frag.deltaPTS = Math.max(deltaPTS, frag.deltaPTS);
    }
    maxStartPTS = Math.max(startPTS, fragStartPts);
    startPTS = Math.min(startPTS, fragStartPts);
    startDTS = Math.min(startDTS, frag.startDTS);
    minEndPTS = Math.min(endPTS, fragEndPts);
    endPTS = Math.max(endPTS, fragEndPts);
    endDTS = Math.max(endDTS, frag.endDTS);
  }
  const drift = startPTS - frag.start;
  if (frag.start !== 0) {
    frag.start = startPTS;
  }
  frag.duration = endPTS - frag.start;
  frag.startPTS = startPTS;
  frag.maxStartPTS = maxStartPTS;
  frag.startDTS = startDTS;
  frag.endPTS = endPTS;
  frag.minEndPTS = minEndPTS;
  frag.endDTS = endDTS;
  const sn = frag.sn;
  if (!details || sn < details.startSN || sn > details.endSN) {
    return 0;
  }
  let i2;
  const fragIdx = sn - details.startSN;
  const fragments = details.fragments;
  fragments[fragIdx] = frag;
  for (i2 = fragIdx; i2 > 0; i2--) {
    updateFromToPTS(fragments[i2], fragments[i2 - 1]);
  }
  for (i2 = fragIdx; i2 < fragments.length - 1; i2++) {
    updateFromToPTS(fragments[i2], fragments[i2 + 1]);
  }
  if (details.fragmentHint) {
    updateFromToPTS(fragments[fragments.length - 1], details.fragmentHint);
  }
  details.PTSKnown = details.alignedSliding = true;
  return drift;
}
function mergeDetails(oldDetails, newDetails) {
  let currentInitSegment = null;
  const oldFragments = oldDetails.fragments;
  for (let i2 = oldFragments.length - 1; i2 >= 0; i2--) {
    const oldInit = oldFragments[i2].initSegment;
    if (oldInit) {
      currentInitSegment = oldInit;
      break;
    }
  }
  if (oldDetails.fragmentHint) {
    delete oldDetails.fragmentHint.endPTS;
  }
  let ccOffset = 0;
  let PTSFrag;
  mapFragmentIntersection(oldDetails, newDetails, (oldFrag, newFrag) => {
    if (oldFrag.relurl) {
      ccOffset = oldFrag.cc - newFrag.cc;
    }
    if (isFiniteNumber(oldFrag.startPTS) && isFiniteNumber(oldFrag.endPTS)) {
      newFrag.start = newFrag.startPTS = oldFrag.startPTS;
      newFrag.startDTS = oldFrag.startDTS;
      newFrag.maxStartPTS = oldFrag.maxStartPTS;
      newFrag.endPTS = oldFrag.endPTS;
      newFrag.endDTS = oldFrag.endDTS;
      newFrag.minEndPTS = oldFrag.minEndPTS;
      newFrag.duration = oldFrag.endPTS - oldFrag.startPTS;
      if (newFrag.duration) {
        PTSFrag = newFrag;
      }
      newDetails.PTSKnown = newDetails.alignedSliding = true;
    }
    newFrag.elementaryStreams = oldFrag.elementaryStreams;
    newFrag.loader = oldFrag.loader;
    newFrag.stats = oldFrag.stats;
    if (oldFrag.initSegment) {
      newFrag.initSegment = oldFrag.initSegment;
      currentInitSegment = oldFrag.initSegment;
    }
  });
  if (currentInitSegment) {
    const fragmentsToCheck = newDetails.fragmentHint ? newDetails.fragments.concat(newDetails.fragmentHint) : newDetails.fragments;
    fragmentsToCheck.forEach((frag) => {
      var _currentInitSegment;
      if (frag && (!frag.initSegment || frag.initSegment.relurl === ((_currentInitSegment = currentInitSegment) == null ? void 0 : _currentInitSegment.relurl))) {
        frag.initSegment = currentInitSegment;
      }
    });
  }
  if (newDetails.skippedSegments) {
    newDetails.deltaUpdateFailed = newDetails.fragments.some((frag) => !frag);
    if (newDetails.deltaUpdateFailed) {
      logger.warn("[level-helper] Previous playlist missing segments skipped in delta playlist");
      for (let i2 = newDetails.skippedSegments; i2--; ) {
        newDetails.fragments.shift();
      }
      newDetails.startSN = newDetails.fragments[0].sn;
      newDetails.startCC = newDetails.fragments[0].cc;
    } else if (newDetails.canSkipDateRanges) {
      newDetails.dateRanges = mergeDateRanges(oldDetails.dateRanges, newDetails.dateRanges, newDetails.recentlyRemovedDateranges);
    }
  }
  const newFragments = newDetails.fragments;
  if (ccOffset) {
    logger.warn("discontinuity sliding from playlist, take drift into account");
    for (let i2 = 0; i2 < newFragments.length; i2++) {
      newFragments[i2].cc += ccOffset;
    }
  }
  if (newDetails.skippedSegments) {
    newDetails.startCC = newDetails.fragments[0].cc;
  }
  mapPartIntersection(oldDetails.partList, newDetails.partList, (oldPart, newPart) => {
    newPart.elementaryStreams = oldPart.elementaryStreams;
    newPart.stats = oldPart.stats;
  });
  if (PTSFrag) {
    updateFragPTSDTS(newDetails, PTSFrag, PTSFrag.startPTS, PTSFrag.endPTS, PTSFrag.startDTS, PTSFrag.endDTS);
  } else {
    adjustSliding(oldDetails, newDetails);
  }
  if (newFragments.length) {
    newDetails.totalduration = newDetails.edge - newFragments[0].start;
  }
  newDetails.driftStartTime = oldDetails.driftStartTime;
  newDetails.driftStart = oldDetails.driftStart;
  const advancedDateTime = newDetails.advancedDateTime;
  if (newDetails.advanced && advancedDateTime) {
    const edge = newDetails.edge;
    if (!newDetails.driftStart) {
      newDetails.driftStartTime = advancedDateTime;
      newDetails.driftStart = edge;
    }
    newDetails.driftEndTime = advancedDateTime;
    newDetails.driftEnd = edge;
  } else {
    newDetails.driftEndTime = oldDetails.driftEndTime;
    newDetails.driftEnd = oldDetails.driftEnd;
    newDetails.advancedDateTime = oldDetails.advancedDateTime;
  }
}
function mergeDateRanges(oldDateRanges, deltaDateRanges, recentlyRemovedDateranges) {
  const dateRanges = _extends({}, oldDateRanges);
  if (recentlyRemovedDateranges) {
    recentlyRemovedDateranges.forEach((id) => {
      delete dateRanges[id];
    });
  }
  Object.keys(deltaDateRanges).forEach((id) => {
    const dateRange = new DateRange(deltaDateRanges[id].attr, dateRanges[id]);
    if (dateRange.isValid) {
      dateRanges[id] = dateRange;
    } else {
      logger.warn(`Ignoring invalid Playlist Delta Update DATERANGE tag: "${JSON.stringify(deltaDateRanges[id].attr)}"`);
    }
  });
  return dateRanges;
}
function mapPartIntersection(oldParts, newParts, intersectionFn) {
  if (oldParts && newParts) {
    let delta = 0;
    for (let i2 = 0, len = oldParts.length; i2 <= len; i2++) {
      const oldPart = oldParts[i2];
      const newPart = newParts[i2 + delta];
      if (oldPart && newPart && oldPart.index === newPart.index && oldPart.fragment.sn === newPart.fragment.sn) {
        intersectionFn(oldPart, newPart);
      } else {
        delta--;
      }
    }
  }
}
function mapFragmentIntersection(oldDetails, newDetails, intersectionFn) {
  const skippedSegments = newDetails.skippedSegments;
  const start = Math.max(oldDetails.startSN, newDetails.startSN) - newDetails.startSN;
  const end = (oldDetails.fragmentHint ? 1 : 0) + (skippedSegments ? newDetails.endSN : Math.min(oldDetails.endSN, newDetails.endSN)) - newDetails.startSN;
  const delta = newDetails.startSN - oldDetails.startSN;
  const newFrags = newDetails.fragmentHint ? newDetails.fragments.concat(newDetails.fragmentHint) : newDetails.fragments;
  const oldFrags = oldDetails.fragmentHint ? oldDetails.fragments.concat(oldDetails.fragmentHint) : oldDetails.fragments;
  for (let i2 = start; i2 <= end; i2++) {
    const oldFrag = oldFrags[delta + i2];
    let newFrag = newFrags[i2];
    if (skippedSegments && !newFrag && i2 < skippedSegments) {
      newFrag = newDetails.fragments[i2] = oldFrag;
    }
    if (oldFrag && newFrag) {
      intersectionFn(oldFrag, newFrag);
    }
  }
}
function adjustSliding(oldDetails, newDetails) {
  const delta = newDetails.startSN + newDetails.skippedSegments - oldDetails.startSN;
  const oldFragments = oldDetails.fragments;
  if (delta < 0 || delta >= oldFragments.length) {
    return;
  }
  addSliding(newDetails, oldFragments[delta].start);
}
function addSliding(details, start) {
  if (start) {
    const fragments = details.fragments;
    for (let i2 = details.skippedSegments; i2 < fragments.length; i2++) {
      fragments[i2].start += start;
    }
    if (details.fragmentHint) {
      details.fragmentHint.start += start;
    }
  }
}
function computeReloadInterval(newDetails, distanceToLiveEdgeMs = Infinity) {
  let reloadInterval = 1e3 * newDetails.targetduration;
  if (newDetails.updated) {
    const fragments = newDetails.fragments;
    const liveEdgeMaxTargetDurations = 4;
    if (fragments.length && reloadInterval * liveEdgeMaxTargetDurations > distanceToLiveEdgeMs) {
      const lastSegmentDuration = fragments[fragments.length - 1].duration * 1e3;
      if (lastSegmentDuration < reloadInterval) {
        reloadInterval = lastSegmentDuration;
      }
    }
  } else {
    reloadInterval /= 2;
  }
  return Math.round(reloadInterval);
}
function getFragmentWithSN(level, sn, fragCurrent) {
  if (!(level != null && level.details)) {
    return null;
  }
  const levelDetails = level.details;
  let fragment = levelDetails.fragments[sn - levelDetails.startSN];
  if (fragment) {
    return fragment;
  }
  fragment = levelDetails.fragmentHint;
  if (fragment && fragment.sn === sn) {
    return fragment;
  }
  if (sn < levelDetails.startSN && fragCurrent && fragCurrent.sn === sn) {
    return fragCurrent;
  }
  return null;
}
function getPartWith(level, sn, partIndex) {
  var _level$details;
  if (!(level != null && level.details)) {
    return null;
  }
  return findPart((_level$details = level.details) == null ? void 0 : _level$details.partList, sn, partIndex);
}
function findPart(partList, sn, partIndex) {
  if (partList) {
    for (let i2 = partList.length; i2--; ) {
      const part = partList[i2];
      if (part.index === partIndex && part.fragment.sn === sn) {
        return part;
      }
    }
  }
  return null;
}
function reassignFragmentLevelIndexes(levels) {
  levels.forEach((level, index) => {
    const {
      details
    } = level;
    if (details != null && details.fragments) {
      details.fragments.forEach((fragment) => {
        fragment.level = index;
      });
    }
  });
}
function isTimeoutError(error) {
  switch (error.details) {
    case ErrorDetails.FRAG_LOAD_TIMEOUT:
    case ErrorDetails.KEY_LOAD_TIMEOUT:
    case ErrorDetails.LEVEL_LOAD_TIMEOUT:
    case ErrorDetails.MANIFEST_LOAD_TIMEOUT:
      return true;
  }
  return false;
}
function getRetryConfig(loadPolicy, error) {
  const isTimeout = isTimeoutError(error);
  return loadPolicy.default[`${isTimeout ? "timeout" : "error"}Retry`];
}
function getRetryDelay(retryConfig, retryCount) {
  const backoffFactor = retryConfig.backoff === "linear" ? 1 : Math.pow(2, retryCount);
  return Math.min(backoffFactor * retryConfig.retryDelayMs, retryConfig.maxRetryDelayMs);
}
function getLoaderConfigWithoutReties(loderConfig) {
  return _objectSpread2(_objectSpread2({}, loderConfig), {
    errorRetry: null,
    timeoutRetry: null
  });
}
function shouldRetry(retryConfig, retryCount, isTimeout, loaderResponse) {
  if (!retryConfig) {
    return false;
  }
  const httpStatus = loaderResponse == null ? void 0 : loaderResponse.code;
  const retry = retryCount < retryConfig.maxNumRetry && (retryForHttpStatus(httpStatus) || !!isTimeout);
  return retryConfig.shouldRetry ? retryConfig.shouldRetry(retryConfig, retryCount, isTimeout, loaderResponse, retry) : retry;
}
function retryForHttpStatus(httpStatus) {
  return httpStatus === 0 && navigator.onLine === false || !!httpStatus && (httpStatus < 400 || httpStatus > 499);
}
const BinarySearch = {
  /**
   * Searches for an item in an array which matches a certain condition.
   * This requires the condition to only match one item in the array,
   * and for the array to be ordered.
   *
   * @param list The array to search.
   * @param comparisonFn
   *      Called and provided a candidate item as the first argument.
   *      Should return:
   *          > -1 if the item should be located at a lower index than the provided item.
   *          > 1 if the item should be located at a higher index than the provided item.
   *          > 0 if the item is the item you're looking for.
   *
   * @returns the object if found, otherwise returns null
   */
  search: function(list, comparisonFn) {
    let minIndex = 0;
    let maxIndex = list.length - 1;
    let currentIndex = null;
    let currentElement = null;
    while (minIndex <= maxIndex) {
      currentIndex = (minIndex + maxIndex) / 2 | 0;
      currentElement = list[currentIndex];
      const comparisonResult = comparisonFn(currentElement);
      if (comparisonResult > 0) {
        minIndex = currentIndex + 1;
      } else if (comparisonResult < 0) {
        maxIndex = currentIndex - 1;
      } else {
        return currentElement;
      }
    }
    return null;
  }
};
function findFragmentByPDT(fragments, PDTValue, maxFragLookUpTolerance) {
  if (PDTValue === null || !Array.isArray(fragments) || !fragments.length || !isFiniteNumber(PDTValue)) {
    return null;
  }
  const startPDT = fragments[0].programDateTime;
  if (PDTValue < (startPDT || 0)) {
    return null;
  }
  const endPDT = fragments[fragments.length - 1].endProgramDateTime;
  if (PDTValue >= (endPDT || 0)) {
    return null;
  }
  maxFragLookUpTolerance = maxFragLookUpTolerance || 0;
  for (let seg = 0; seg < fragments.length; ++seg) {
    const frag = fragments[seg];
    if (pdtWithinToleranceTest(PDTValue, maxFragLookUpTolerance, frag)) {
      return frag;
    }
  }
  return null;
}
function findFragmentByPTS(fragPrevious, fragments, bufferEnd = 0, maxFragLookUpTolerance = 0, nextFragLookupTolerance = 5e-3) {
  let fragNext = null;
  if (fragPrevious) {
    fragNext = fragments[fragPrevious.sn - fragments[0].sn + 1] || null;
    const bufferEdgeError = fragPrevious.endDTS - bufferEnd;
    if (bufferEdgeError > 0 && bufferEdgeError < 15e-7) {
      bufferEnd += 15e-7;
    }
  } else if (bufferEnd === 0 && fragments[0].start === 0) {
    fragNext = fragments[0];
  }
  if (fragNext && ((!fragPrevious || fragPrevious.level === fragNext.level) && fragmentWithinToleranceTest(bufferEnd, maxFragLookUpTolerance, fragNext) === 0 || fragmentWithinFastStartSwitch(fragNext, fragPrevious, Math.min(nextFragLookupTolerance, maxFragLookUpTolerance)))) {
    return fragNext;
  }
  const foundFragment = BinarySearch.search(fragments, fragmentWithinToleranceTest.bind(null, bufferEnd, maxFragLookUpTolerance));
  if (foundFragment && (foundFragment !== fragPrevious || !fragNext)) {
    return foundFragment;
  }
  return fragNext;
}
function fragmentWithinFastStartSwitch(fragNext, fragPrevious, nextFragLookupTolerance) {
  if (fragPrevious && fragPrevious.start === 0 && fragPrevious.level < fragNext.level && (fragPrevious.endPTS || 0) > 0) {
    const firstDuration = fragPrevious.tagList.reduce((duration, tag) => {
      if (tag[0] === "INF") {
        duration += parseFloat(tag[1]);
      }
      return duration;
    }, nextFragLookupTolerance);
    return fragNext.start <= firstDuration;
  }
  return false;
}
function fragmentWithinToleranceTest(bufferEnd = 0, maxFragLookUpTolerance = 0, candidate) {
  if (candidate.start <= bufferEnd && candidate.start + candidate.duration > bufferEnd) {
    return 0;
  }
  const candidateLookupTolerance = Math.min(maxFragLookUpTolerance, candidate.duration + (candidate.deltaPTS ? candidate.deltaPTS : 0));
  if (candidate.start + candidate.duration - candidateLookupTolerance <= bufferEnd) {
    return 1;
  } else if (candidate.start - candidateLookupTolerance > bufferEnd && candidate.start) {
    return -1;
  }
  return 0;
}
function pdtWithinToleranceTest(pdtBufferEnd, maxFragLookUpTolerance, candidate) {
  const candidateLookupTolerance = Math.min(maxFragLookUpTolerance, candidate.duration + (candidate.deltaPTS ? candidate.deltaPTS : 0)) * 1e3;
  const endProgramDateTime = candidate.endProgramDateTime || 0;
  return endProgramDateTime - candidateLookupTolerance > pdtBufferEnd;
}
function findFragWithCC(fragments, cc) {
  return BinarySearch.search(fragments, (candidate) => {
    if (candidate.cc < cc) {
      return 1;
    } else if (candidate.cc > cc) {
      return -1;
    } else {
      return 0;
    }
  });
}
var NetworkErrorAction = {
  DoNothing: 0,
  SendAlternateToPenaltyBox: 2,
  RemoveAlternatePermanently: 3,
  RetryRequest: 5
};
var ErrorActionFlags = {
  None: 0,
  MoveAllAlternatesMatchingHost: 1,
  MoveAllAlternatesMatchingHDCP: 2
};
class ErrorController {
  constructor(hls) {
    this.hls = void 0;
    this.playlistError = 0;
    this.penalizedRenditions = {};
    this.log = void 0;
    this.warn = void 0;
    this.error = void 0;
    this.hls = hls;
    this.log = logger.log.bind(logger, `[info]:`);
    this.warn = logger.warn.bind(logger, `[warning]:`);
    this.error = logger.error.bind(logger, `[error]:`);
    this.registerListeners();
  }
  registerListeners() {
    const hls = this.hls;
    hls.on(Events.ERROR, this.onError, this);
    hls.on(Events.MANIFEST_LOADING, this.onManifestLoading, this);
    hls.on(Events.LEVEL_UPDATED, this.onLevelUpdated, this);
  }
  unregisterListeners() {
    const hls = this.hls;
    if (!hls) {
      return;
    }
    hls.off(Events.ERROR, this.onError, this);
    hls.off(Events.ERROR, this.onErrorOut, this);
    hls.off(Events.MANIFEST_LOADING, this.onManifestLoading, this);
    hls.off(Events.LEVEL_UPDATED, this.onLevelUpdated, this);
  }
  destroy() {
    this.unregisterListeners();
    this.hls = null;
    this.penalizedRenditions = {};
  }
  startLoad(startPosition) {
  }
  stopLoad() {
    this.playlistError = 0;
  }
  getVariantLevelIndex(frag) {
    return (frag == null ? void 0 : frag.type) === PlaylistLevelType.MAIN ? frag.level : this.hls.loadLevel;
  }
  onManifestLoading() {
    this.playlistError = 0;
    this.penalizedRenditions = {};
  }
  onLevelUpdated() {
    this.playlistError = 0;
  }
  onError(event, data) {
    var _data$frag, _data$level;
    if (data.fatal) {
      return;
    }
    const hls = this.hls;
    const context = data.context;
    switch (data.details) {
      case ErrorDetails.FRAG_LOAD_ERROR:
      case ErrorDetails.FRAG_LOAD_TIMEOUT:
      case ErrorDetails.KEY_LOAD_ERROR:
      case ErrorDetails.KEY_LOAD_TIMEOUT:
        data.errorAction = this.getFragRetryOrSwitchAction(data);
        return;
      case ErrorDetails.FRAG_PARSING_ERROR:
        if ((_data$frag = data.frag) != null && _data$frag.gap) {
          data.errorAction = {
            action: NetworkErrorAction.DoNothing,
            flags: ErrorActionFlags.None
          };
          return;
        }
      // falls through
      case ErrorDetails.FRAG_GAP:
      case ErrorDetails.FRAG_DECRYPT_ERROR: {
        data.errorAction = this.getFragRetryOrSwitchAction(data);
        data.errorAction.action = NetworkErrorAction.SendAlternateToPenaltyBox;
        return;
      }
      case ErrorDetails.LEVEL_EMPTY_ERROR:
      case ErrorDetails.LEVEL_PARSING_ERROR:
        {
          var _data$context, _data$context$levelDe;
          const levelIndex = data.parent === PlaylistLevelType.MAIN ? data.level : hls.loadLevel;
          if (data.details === ErrorDetails.LEVEL_EMPTY_ERROR && !!((_data$context = data.context) != null && (_data$context$levelDe = _data$context.levelDetails) != null && _data$context$levelDe.live)) {
            data.errorAction = this.getPlaylistRetryOrSwitchAction(data, levelIndex);
          } else {
            data.levelRetry = false;
            data.errorAction = this.getLevelSwitchAction(data, levelIndex);
          }
        }
        return;
      case ErrorDetails.LEVEL_LOAD_ERROR:
      case ErrorDetails.LEVEL_LOAD_TIMEOUT:
        if (typeof (context == null ? void 0 : context.level) === "number") {
          data.errorAction = this.getPlaylistRetryOrSwitchAction(data, context.level);
        }
        return;
      case ErrorDetails.AUDIO_TRACK_LOAD_ERROR:
      case ErrorDetails.AUDIO_TRACK_LOAD_TIMEOUT:
      case ErrorDetails.SUBTITLE_LOAD_ERROR:
      case ErrorDetails.SUBTITLE_TRACK_LOAD_TIMEOUT:
        if (context) {
          const level = hls.levels[hls.loadLevel];
          if (level && (context.type === PlaylistContextType.AUDIO_TRACK && level.hasAudioGroup(context.groupId) || context.type === PlaylistContextType.SUBTITLE_TRACK && level.hasSubtitleGroup(context.groupId))) {
            data.errorAction = this.getPlaylistRetryOrSwitchAction(data, hls.loadLevel);
            data.errorAction.action = NetworkErrorAction.SendAlternateToPenaltyBox;
            data.errorAction.flags = ErrorActionFlags.MoveAllAlternatesMatchingHost;
            return;
          }
        }
        return;
      case ErrorDetails.KEY_SYSTEM_STATUS_OUTPUT_RESTRICTED:
        {
          const level = hls.levels[hls.loadLevel];
          const restrictedHdcpLevel = level == null ? void 0 : level.attrs["HDCP-LEVEL"];
          if (restrictedHdcpLevel) {
            data.errorAction = {
              action: NetworkErrorAction.SendAlternateToPenaltyBox,
              flags: ErrorActionFlags.MoveAllAlternatesMatchingHDCP,
              hdcpLevel: restrictedHdcpLevel
            };
          } else {
            this.keySystemError(data);
          }
        }
        return;
      case ErrorDetails.BUFFER_ADD_CODEC_ERROR:
      case ErrorDetails.REMUX_ALLOC_ERROR:
      case ErrorDetails.BUFFER_APPEND_ERROR:
        data.errorAction = this.getLevelSwitchAction(data, (_data$level = data.level) != null ? _data$level : hls.loadLevel);
        return;
      case ErrorDetails.INTERNAL_EXCEPTION:
      case ErrorDetails.BUFFER_APPENDING_ERROR:
      case ErrorDetails.BUFFER_FULL_ERROR:
      case ErrorDetails.LEVEL_SWITCH_ERROR:
      case ErrorDetails.BUFFER_STALLED_ERROR:
      case ErrorDetails.BUFFER_SEEK_OVER_HOLE:
      case ErrorDetails.BUFFER_NUDGE_ON_STALL:
        data.errorAction = {
          action: NetworkErrorAction.DoNothing,
          flags: ErrorActionFlags.None
        };
        return;
    }
    if (data.type === ErrorTypes.KEY_SYSTEM_ERROR) {
      this.keySystemError(data);
    }
  }
  keySystemError(data) {
    const levelIndex = this.getVariantLevelIndex(data.frag);
    data.levelRetry = false;
    data.errorAction = this.getLevelSwitchAction(data, levelIndex);
  }
  getPlaylistRetryOrSwitchAction(data, levelIndex) {
    const hls = this.hls;
    const retryConfig = getRetryConfig(hls.config.playlistLoadPolicy, data);
    const retryCount = this.playlistError++;
    const retry = shouldRetry(retryConfig, retryCount, isTimeoutError(data), data.response);
    if (retry) {
      return {
        action: NetworkErrorAction.RetryRequest,
        flags: ErrorActionFlags.None,
        retryConfig,
        retryCount
      };
    }
    const errorAction = this.getLevelSwitchAction(data, levelIndex);
    if (retryConfig) {
      errorAction.retryConfig = retryConfig;
      errorAction.retryCount = retryCount;
    }
    return errorAction;
  }
  getFragRetryOrSwitchAction(data) {
    const hls = this.hls;
    const variantLevelIndex = this.getVariantLevelIndex(data.frag);
    const level = hls.levels[variantLevelIndex];
    const {
      fragLoadPolicy,
      keyLoadPolicy
    } = hls.config;
    const retryConfig = getRetryConfig(data.details.startsWith("key") ? keyLoadPolicy : fragLoadPolicy, data);
    const fragmentErrors = hls.levels.reduce((acc, level2) => acc + level2.fragmentError, 0);
    if (level) {
      if (data.details !== ErrorDetails.FRAG_GAP) {
        level.fragmentError++;
      }
      const retry = shouldRetry(retryConfig, fragmentErrors, isTimeoutError(data), data.response);
      if (retry) {
        return {
          action: NetworkErrorAction.RetryRequest,
          flags: ErrorActionFlags.None,
          retryConfig,
          retryCount: fragmentErrors
        };
      }
    }
    const errorAction = this.getLevelSwitchAction(data, variantLevelIndex);
    if (retryConfig) {
      errorAction.retryConfig = retryConfig;
      errorAction.retryCount = fragmentErrors;
    }
    return errorAction;
  }
  getLevelSwitchAction(data, levelIndex) {
    const hls = this.hls;
    if (levelIndex === null || levelIndex === void 0) {
      levelIndex = hls.loadLevel;
    }
    const level = this.hls.levels[levelIndex];
    if (level) {
      var _data$frag2, _data$context2;
      const errorDetails = data.details;
      level.loadError++;
      if (errorDetails === ErrorDetails.BUFFER_APPEND_ERROR) {
        level.fragmentError++;
      }
      let nextLevel = -1;
      const {
        levels,
        loadLevel,
        minAutoLevel,
        maxAutoLevel
      } = hls;
      if (!hls.autoLevelEnabled) {
        hls.loadLevel = -1;
      }
      const fragErrorType = (_data$frag2 = data.frag) == null ? void 0 : _data$frag2.type;
      const isAudioCodecError = fragErrorType === PlaylistLevelType.AUDIO && errorDetails === ErrorDetails.FRAG_PARSING_ERROR || data.sourceBufferName === "audio" && (errorDetails === ErrorDetails.BUFFER_ADD_CODEC_ERROR || errorDetails === ErrorDetails.BUFFER_APPEND_ERROR);
      const findAudioCodecAlternate = isAudioCodecError && levels.some(({
        audioCodec
      }) => level.audioCodec !== audioCodec);
      const isVideoCodecError = data.sourceBufferName === "video" && (errorDetails === ErrorDetails.BUFFER_ADD_CODEC_ERROR || errorDetails === ErrorDetails.BUFFER_APPEND_ERROR);
      const findVideoCodecAlternate = isVideoCodecError && levels.some(({
        codecSet,
        audioCodec
      }) => level.codecSet !== codecSet && level.audioCodec === audioCodec);
      const {
        type: playlistErrorType,
        groupId: playlistErrorGroupId
      } = (_data$context2 = data.context) != null ? _data$context2 : {};
      for (let i2 = levels.length; i2--; ) {
        const candidate = (i2 + loadLevel) % levels.length;
        if (candidate !== loadLevel && candidate >= minAutoLevel && candidate <= maxAutoLevel && levels[candidate].loadError === 0) {
          var _level$audioGroups, _level$subtitleGroups;
          const levelCandidate = levels[candidate];
          if (errorDetails === ErrorDetails.FRAG_GAP && fragErrorType === PlaylistLevelType.MAIN && data.frag) {
            const levelDetails = levels[candidate].details;
            if (levelDetails) {
              const fragCandidate = findFragmentByPTS(data.frag, levelDetails.fragments, data.frag.start);
              if (fragCandidate != null && fragCandidate.gap) {
                continue;
              }
            }
          } else if (playlistErrorType === PlaylistContextType.AUDIO_TRACK && levelCandidate.hasAudioGroup(playlistErrorGroupId) || playlistErrorType === PlaylistContextType.SUBTITLE_TRACK && levelCandidate.hasSubtitleGroup(playlistErrorGroupId)) {
            continue;
          } else if (fragErrorType === PlaylistLevelType.AUDIO && (_level$audioGroups = level.audioGroups) != null && _level$audioGroups.some((groupId) => levelCandidate.hasAudioGroup(groupId)) || fragErrorType === PlaylistLevelType.SUBTITLE && (_level$subtitleGroups = level.subtitleGroups) != null && _level$subtitleGroups.some((groupId) => levelCandidate.hasSubtitleGroup(groupId)) || findAudioCodecAlternate && level.audioCodec === levelCandidate.audioCodec || !findAudioCodecAlternate && level.audioCodec !== levelCandidate.audioCodec || findVideoCodecAlternate && level.codecSet === levelCandidate.codecSet) {
            continue;
          }
          nextLevel = candidate;
          break;
        }
      }
      if (nextLevel > -1 && hls.loadLevel !== nextLevel) {
        data.levelRetry = true;
        this.playlistError = 0;
        return {
          action: NetworkErrorAction.SendAlternateToPenaltyBox,
          flags: ErrorActionFlags.None,
          nextAutoLevel: nextLevel
        };
      }
    }
    return {
      action: NetworkErrorAction.SendAlternateToPenaltyBox,
      flags: ErrorActionFlags.MoveAllAlternatesMatchingHost
    };
  }
  onErrorOut(event, data) {
    var _data$errorAction;
    switch ((_data$errorAction = data.errorAction) == null ? void 0 : _data$errorAction.action) {
      case NetworkErrorAction.DoNothing:
        break;
      case NetworkErrorAction.SendAlternateToPenaltyBox:
        this.sendAlternateToPenaltyBox(data);
        if (!data.errorAction.resolved && data.details !== ErrorDetails.FRAG_GAP) {
          data.fatal = true;
        } else if (/MediaSource readyState: ended/.test(data.error.message)) {
          this.warn(`MediaSource ended after "${data.sourceBufferName}" sourceBuffer append error. Attempting to recover from media error.`);
          this.hls.recoverMediaError();
        }
        break;
    }
    if (data.fatal) {
      this.hls.stopLoad();
      return;
    }
  }
  sendAlternateToPenaltyBox(data) {
    const hls = this.hls;
    const errorAction = data.errorAction;
    if (!errorAction) {
      return;
    }
    const {
      flags,
      hdcpLevel,
      nextAutoLevel
    } = errorAction;
    switch (flags) {
      case ErrorActionFlags.None:
        this.switchLevel(data, nextAutoLevel);
        break;
      case ErrorActionFlags.MoveAllAlternatesMatchingHDCP:
        if (hdcpLevel) {
          hls.maxHdcpLevel = HdcpLevels[HdcpLevels.indexOf(hdcpLevel) - 1];
          errorAction.resolved = true;
        }
        this.warn(`Restricting playback to HDCP-LEVEL of "${hls.maxHdcpLevel}" or lower`);
        break;
    }
    if (!errorAction.resolved) {
      this.switchLevel(data, nextAutoLevel);
    }
  }
  switchLevel(data, levelIndex) {
    if (levelIndex !== void 0 && data.errorAction) {
      this.warn(`switching to level ${levelIndex} after ${data.details}`);
      this.hls.nextAutoLevel = levelIndex;
      data.errorAction.resolved = true;
      this.hls.nextLoadLevel = this.hls.nextAutoLevel;
    }
  }
}
class BasePlaylistController {
  constructor(hls, logPrefix) {
    this.hls = void 0;
    this.timer = -1;
    this.requestScheduled = -1;
    this.canLoad = false;
    this.log = void 0;
    this.warn = void 0;
    this.log = logger.log.bind(logger, `${logPrefix}:`);
    this.warn = logger.warn.bind(logger, `${logPrefix}:`);
    this.hls = hls;
  }
  destroy() {
    this.clearTimer();
    this.hls = this.log = this.warn = null;
  }
  clearTimer() {
    if (this.timer !== -1) {
      self.clearTimeout(this.timer);
      this.timer = -1;
    }
  }
  startLoad() {
    this.canLoad = true;
    this.requestScheduled = -1;
    this.loadPlaylist();
  }
  stopLoad() {
    this.canLoad = false;
    this.clearTimer();
  }
  switchParams(playlistUri, previous, current) {
    const renditionReports = previous == null ? void 0 : previous.renditionReports;
    if (renditionReports) {
      let foundIndex = -1;
      for (let i2 = 0; i2 < renditionReports.length; i2++) {
        const attr = renditionReports[i2];
        let uri;
        try {
          uri = new self.URL(attr.URI, previous.url).href;
        } catch (error) {
          logger.warn(`Could not construct new URL for Rendition Report: ${error}`);
          uri = attr.URI || "";
        }
        if (uri === playlistUri) {
          foundIndex = i2;
          break;
        } else if (uri === playlistUri.substring(0, uri.length)) {
          foundIndex = i2;
        }
      }
      if (foundIndex !== -1) {
        const attr = renditionReports[foundIndex];
        const msn = parseInt(attr["LAST-MSN"]) || (previous == null ? void 0 : previous.lastPartSn);
        let part = parseInt(attr["LAST-PART"]) || (previous == null ? void 0 : previous.lastPartIndex);
        if (this.hls.config.lowLatencyMode) {
          const currentGoal = Math.min(previous.age - previous.partTarget, previous.targetduration);
          if (part >= 0 && currentGoal > previous.partTarget) {
            part += 1;
          }
        }
        const skip = current && getSkipValue(current);
        return new HlsUrlParameters(msn, part >= 0 ? part : void 0, skip);
      }
    }
  }
  loadPlaylist(hlsUrlParameters) {
    if (this.requestScheduled === -1) {
      this.requestScheduled = self.performance.now();
    }
  }
  shouldLoadPlaylist(playlist) {
    return this.canLoad && !!playlist && !!playlist.url && (!playlist.details || playlist.details.live);
  }
  shouldReloadPlaylist(playlist) {
    return this.timer === -1 && this.requestScheduled === -1 && this.shouldLoadPlaylist(playlist);
  }
  playlistLoaded(index, data, previousDetails) {
    const {
      details,
      stats
    } = data;
    const now2 = self.performance.now();
    const elapsed = stats.loading.first ? Math.max(0, now2 - stats.loading.first) : 0;
    details.advancedDateTime = Date.now() - elapsed;
    if (details.live || previousDetails != null && previousDetails.live) {
      details.reloaded(previousDetails);
      if (previousDetails) {
        this.log(`live playlist ${index} ${details.advanced ? "REFRESHED " + details.lastPartSn + "-" + details.lastPartIndex : details.updated ? "UPDATED" : "MISSED"}`);
      }
      if (previousDetails && details.fragments.length > 0) {
        mergeDetails(previousDetails, details);
      }
      if (!this.canLoad || !details.live) {
        return;
      }
      let deliveryDirectives;
      let msn = void 0;
      let part = void 0;
      if (details.canBlockReload && details.endSN && details.advanced) {
        const lowLatencyMode = this.hls.config.lowLatencyMode;
        const lastPartSn = details.lastPartSn;
        const endSn = details.endSN;
        const lastPartIndex = details.lastPartIndex;
        const hasParts = lastPartIndex !== -1;
        const lastPart = lastPartSn === endSn;
        const nextSnStartIndex = lowLatencyMode ? 0 : lastPartIndex;
        if (hasParts) {
          msn = lastPart ? endSn + 1 : lastPartSn;
          part = lastPart ? nextSnStartIndex : lastPartIndex + 1;
        } else {
          msn = endSn + 1;
        }
        const lastAdvanced = details.age;
        const cdnAge = lastAdvanced + details.ageHeader;
        let currentGoal = Math.min(cdnAge - details.partTarget, details.targetduration * 1.5);
        if (currentGoal > 0) {
          if (previousDetails && currentGoal > previousDetails.tuneInGoal) {
            this.warn(`CDN Tune-in goal increased from: ${previousDetails.tuneInGoal} to: ${currentGoal} with playlist age: ${details.age}`);
            currentGoal = 0;
          } else {
            const segments = Math.floor(currentGoal / details.targetduration);
            msn += segments;
            if (part !== void 0) {
              const parts = Math.round(currentGoal % details.targetduration / details.partTarget);
              part += parts;
            }
            this.log(`CDN Tune-in age: ${details.ageHeader}s last advanced ${lastAdvanced.toFixed(2)}s goal: ${currentGoal} skip sn ${segments} to part ${part}`);
          }
          details.tuneInGoal = currentGoal;
        }
        deliveryDirectives = this.getDeliveryDirectives(details, data.deliveryDirectives, msn, part);
        if (lowLatencyMode || !lastPart) {
          this.loadPlaylist(deliveryDirectives);
          return;
        }
      } else if (details.canBlockReload || details.canSkipUntil) {
        deliveryDirectives = this.getDeliveryDirectives(details, data.deliveryDirectives, msn, part);
      }
      const bufferInfo = this.hls.mainForwardBufferInfo;
      const position = bufferInfo ? bufferInfo.end - bufferInfo.len : 0;
      const distanceToLiveEdgeMs = (details.edge - position) * 1e3;
      const reloadInterval = computeReloadInterval(details, distanceToLiveEdgeMs);
      if (details.updated && now2 > this.requestScheduled + reloadInterval) {
        this.requestScheduled = stats.loading.start;
      }
      if (msn !== void 0 && details.canBlockReload) {
        this.requestScheduled = stats.loading.first + reloadInterval - (details.partTarget * 1e3 || 1e3);
      } else if (this.requestScheduled === -1 || this.requestScheduled + reloadInterval < now2) {
        this.requestScheduled = now2;
      } else if (this.requestScheduled - now2 <= 0) {
        this.requestScheduled += reloadInterval;
      }
      let estimatedTimeUntilUpdate = this.requestScheduled - now2;
      estimatedTimeUntilUpdate = Math.max(0, estimatedTimeUntilUpdate);
      this.log(`reload live playlist ${index} in ${Math.round(estimatedTimeUntilUpdate)} ms`);
      this.timer = self.setTimeout(() => this.loadPlaylist(deliveryDirectives), estimatedTimeUntilUpdate);
    } else {
      this.clearTimer();
    }
  }
  getDeliveryDirectives(details, previousDeliveryDirectives, msn, part) {
    let skip = getSkipValue(details);
    if (previousDeliveryDirectives != null && previousDeliveryDirectives.skip && details.deltaUpdateFailed) {
      msn = previousDeliveryDirectives.msn;
      part = previousDeliveryDirectives.part;
      skip = HlsSkip.No;
    }
    return new HlsUrlParameters(msn, part, skip);
  }
  checkRetry(errorEvent) {
    const errorDetails = errorEvent.details;
    const isTimeout = isTimeoutError(errorEvent);
    const errorAction = errorEvent.errorAction;
    const {
      action,
      retryCount = 0,
      retryConfig
    } = errorAction || {};
    const retry = !!errorAction && !!retryConfig && (action === NetworkErrorAction.RetryRequest || !errorAction.resolved && action === NetworkErrorAction.SendAlternateToPenaltyBox);
    if (retry) {
      var _errorEvent$context;
      this.requestScheduled = -1;
      if (retryCount >= retryConfig.maxNumRetry) {
        return false;
      }
      if (isTimeout && (_errorEvent$context = errorEvent.context) != null && _errorEvent$context.deliveryDirectives) {
        this.warn(`Retrying playlist loading ${retryCount + 1}/${retryConfig.maxNumRetry} after "${errorDetails}" without delivery-directives`);
        this.loadPlaylist();
      } else {
        const delay = getRetryDelay(retryConfig, retryCount);
        this.timer = self.setTimeout(() => this.loadPlaylist(), delay);
        this.warn(`Retrying playlist loading ${retryCount + 1}/${retryConfig.maxNumRetry} after "${errorDetails}" in ${delay}ms`);
      }
      errorEvent.levelRetry = true;
      errorAction.resolved = true;
    }
    return retry;
  }
}
class EWMA {
  //  About half of the estimated value will be from the last |halfLife| samples by weight.
  constructor(halfLife, estimate = 0, weight = 0) {
    this.halfLife = void 0;
    this.alpha_ = void 0;
    this.estimate_ = void 0;
    this.totalWeight_ = void 0;
    this.halfLife = halfLife;
    this.alpha_ = halfLife ? Math.exp(Math.log(0.5) / halfLife) : 0;
    this.estimate_ = estimate;
    this.totalWeight_ = weight;
  }
  sample(weight, value) {
    const adjAlpha = Math.pow(this.alpha_, weight);
    this.estimate_ = value * (1 - adjAlpha) + adjAlpha * this.estimate_;
    this.totalWeight_ += weight;
  }
  getTotalWeight() {
    return this.totalWeight_;
  }
  getEstimate() {
    if (this.alpha_) {
      const zeroFactor = 1 - Math.pow(this.alpha_, this.totalWeight_);
      if (zeroFactor) {
        return this.estimate_ / zeroFactor;
      }
    }
    return this.estimate_;
  }
}
class EwmaBandWidthEstimator {
  constructor(slow, fast, defaultEstimate, defaultTTFB = 100) {
    this.defaultEstimate_ = void 0;
    this.minWeight_ = void 0;
    this.minDelayMs_ = void 0;
    this.slow_ = void 0;
    this.fast_ = void 0;
    this.defaultTTFB_ = void 0;
    this.ttfb_ = void 0;
    this.defaultEstimate_ = defaultEstimate;
    this.minWeight_ = 1e-3;
    this.minDelayMs_ = 50;
    this.slow_ = new EWMA(slow);
    this.fast_ = new EWMA(fast);
    this.defaultTTFB_ = defaultTTFB;
    this.ttfb_ = new EWMA(slow);
  }
  update(slow, fast) {
    const {
      slow_,
      fast_,
      ttfb_
    } = this;
    if (slow_.halfLife !== slow) {
      this.slow_ = new EWMA(slow, slow_.getEstimate(), slow_.getTotalWeight());
    }
    if (fast_.halfLife !== fast) {
      this.fast_ = new EWMA(fast, fast_.getEstimate(), fast_.getTotalWeight());
    }
    if (ttfb_.halfLife !== slow) {
      this.ttfb_ = new EWMA(slow, ttfb_.getEstimate(), ttfb_.getTotalWeight());
    }
  }
  sample(durationMs, numBytes) {
    durationMs = Math.max(durationMs, this.minDelayMs_);
    const numBits = 8 * numBytes;
    const durationS = durationMs / 1e3;
    const bandwidthInBps = numBits / durationS;
    this.fast_.sample(durationS, bandwidthInBps);
    this.slow_.sample(durationS, bandwidthInBps);
  }
  sampleTTFB(ttfb) {
    const seconds = ttfb / 1e3;
    const weight = Math.sqrt(2) * Math.exp(-Math.pow(seconds, 2) / 2);
    this.ttfb_.sample(weight, Math.max(ttfb, 5));
  }
  canEstimate() {
    return this.fast_.getTotalWeight() >= this.minWeight_;
  }
  getEstimate() {
    if (this.canEstimate()) {
      return Math.min(this.fast_.getEstimate(), this.slow_.getEstimate());
    } else {
      return this.defaultEstimate_;
    }
  }
  getEstimateTTFB() {
    if (this.ttfb_.getTotalWeight() >= this.minWeight_) {
      return this.ttfb_.getEstimate();
    } else {
      return this.defaultTTFB_;
    }
  }
  destroy() {
  }
}
const SUPPORTED_INFO_DEFAULT = {
  supported: true,
  configurations: [],
  decodingInfoResults: [{
    supported: true,
    powerEfficient: true,
    smooth: true
  }]
};
const SUPPORTED_INFO_CACHE = {};
function requiresMediaCapabilitiesDecodingInfo(level, audioTracksByGroup, currentVideoRange, currentFrameRate, currentBw, audioPreference) {
  const audioGroups = level.audioCodec ? level.audioGroups : null;
  const audioCodecPreference = audioPreference == null ? void 0 : audioPreference.audioCodec;
  const channelsPreference = audioPreference == null ? void 0 : audioPreference.channels;
  const maxChannels = channelsPreference ? parseInt(channelsPreference) : audioCodecPreference ? Infinity : 2;
  let audioChannels = null;
  if (audioGroups != null && audioGroups.length) {
    try {
      if (audioGroups.length === 1 && audioGroups[0]) {
        audioChannels = audioTracksByGroup.groups[audioGroups[0]].channels;
      } else {
        audioChannels = audioGroups.reduce((acc, groupId) => {
          if (groupId) {
            const audioTrackGroup = audioTracksByGroup.groups[groupId];
            if (!audioTrackGroup) {
              throw new Error(`Audio track group ${groupId} not found`);
            }
            Object.keys(audioTrackGroup.channels).forEach((key) => {
              acc[key] = (acc[key] || 0) + audioTrackGroup.channels[key];
            });
          }
          return acc;
        }, {
          2: 0
        });
      }
    } catch (error) {
      return true;
    }
  }
  return level.videoCodec !== void 0 && (level.width > 1920 && level.height > 1088 || level.height > 1920 && level.width > 1088 || level.frameRate > Math.max(currentFrameRate, 30) || level.videoRange !== "SDR" && level.videoRange !== currentVideoRange || level.bitrate > Math.max(currentBw, 8e6)) || !!audioChannels && isFiniteNumber(maxChannels) && Object.keys(audioChannels).some((channels) => parseInt(channels) > maxChannels);
}
function getMediaDecodingInfoPromise(level, audioTracksByGroup, mediaCapabilities) {
  const videoCodecs = level.videoCodec;
  const audioCodecs = level.audioCodec;
  if (!videoCodecs || !audioCodecs || !mediaCapabilities) {
    return Promise.resolve(SUPPORTED_INFO_DEFAULT);
  }
  const baseVideoConfiguration = {
    width: level.width,
    height: level.height,
    bitrate: Math.ceil(Math.max(level.bitrate * 0.9, level.averageBitrate)),
    // Assume a framerate of 30fps since MediaCapabilities will not accept Level default of 0.
    framerate: level.frameRate || 30
  };
  const videoRange = level.videoRange;
  if (videoRange !== "SDR") {
    baseVideoConfiguration.transferFunction = videoRange.toLowerCase();
  }
  const configurations = videoCodecs.split(",").map((videoCodec) => ({
    type: "media-source",
    video: _objectSpread2(_objectSpread2({}, baseVideoConfiguration), {}, {
      contentType: mimeTypeForCodec(videoCodec, "video")
    })
  }));
  if (audioCodecs && level.audioGroups) {
    level.audioGroups.forEach((audioGroupId) => {
      var _audioTracksByGroup$g;
      if (!audioGroupId) {
        return;
      }
      (_audioTracksByGroup$g = audioTracksByGroup.groups[audioGroupId]) == null ? void 0 : _audioTracksByGroup$g.tracks.forEach((audioTrack) => {
        if (audioTrack.groupId === audioGroupId) {
          const channels = audioTrack.channels || "";
          const channelsNumber = parseFloat(channels);
          if (isFiniteNumber(channelsNumber) && channelsNumber > 2) {
            configurations.push.apply(configurations, audioCodecs.split(",").map((audioCodec) => ({
              type: "media-source",
              audio: {
                contentType: mimeTypeForCodec(audioCodec, "audio"),
                channels: "" + channelsNumber
                // spatialRendering:
                //   audioCodec === 'ec-3' && channels.indexOf('JOC'),
              }
            })));
          }
        }
      });
    });
  }
  return Promise.all(configurations.map((configuration) => {
    const decodingInfoKey = getMediaDecodingInfoKey(configuration);
    return SUPPORTED_INFO_CACHE[decodingInfoKey] || (SUPPORTED_INFO_CACHE[decodingInfoKey] = mediaCapabilities.decodingInfo(configuration));
  })).then((decodingInfoResults) => ({
    supported: !decodingInfoResults.some((info) => !info.supported),
    configurations,
    decodingInfoResults
  })).catch((error) => ({
    supported: false,
    configurations,
    decodingInfoResults: [],
    error
  }));
}
function getMediaDecodingInfoKey(config) {
  const {
    audio,
    video
  } = config;
  const mediaConfig = video || audio;
  if (mediaConfig) {
    const codec = mediaConfig.contentType.split('"')[1];
    if (video) {
      return `r${video.height}x${video.width}f${Math.ceil(video.framerate)}${video.transferFunction || "sd"}_${codec}_${Math.ceil(video.bitrate / 1e5)}`;
    }
    if (audio) {
      return `c${audio.channels}${audio.spatialRendering ? "s" : "n"}_${codec}`;
    }
  }
  return "";
}
function isHdrSupported() {
  if (typeof matchMedia === "function") {
    const mediaQueryList = matchMedia("(dynamic-range: high)");
    const badQuery = matchMedia("bad query");
    if (mediaQueryList.media !== badQuery.media) {
      return mediaQueryList.matches === true;
    }
  }
  return false;
}
function getVideoSelectionOptions(currentVideoRange, videoPreference) {
  let preferHDR = false;
  let allowedVideoRanges = [];
  if (currentVideoRange) {
    preferHDR = currentVideoRange !== "SDR";
    allowedVideoRanges = [currentVideoRange];
  }
  if (videoPreference) {
    allowedVideoRanges = videoPreference.allowedVideoRanges || VideoRangeValues.slice(0);
    preferHDR = videoPreference.preferHDR !== void 0 ? videoPreference.preferHDR : isHdrSupported();
    if (preferHDR) {
      allowedVideoRanges = allowedVideoRanges.filter((range) => range !== "SDR");
    } else {
      allowedVideoRanges = ["SDR"];
    }
  }
  return {
    preferHDR,
    allowedVideoRanges
  };
}
function getStartCodecTier(codecTiers, currentVideoRange, currentBw, audioPreference, videoPreference) {
  const codecSets = Object.keys(codecTiers);
  const channelsPreference = audioPreference == null ? void 0 : audioPreference.channels;
  const audioCodecPreference = audioPreference == null ? void 0 : audioPreference.audioCodec;
  const preferStereo = channelsPreference && parseInt(channelsPreference) === 2;
  let hasStereo = true;
  let hasCurrentVideoRange = false;
  let minHeight = Infinity;
  let minFramerate = Infinity;
  let minBitrate = Infinity;
  let selectedScore = 0;
  let videoRanges = [];
  const {
    preferHDR,
    allowedVideoRanges
  } = getVideoSelectionOptions(currentVideoRange, videoPreference);
  for (let i2 = codecSets.length; i2--; ) {
    const tier = codecTiers[codecSets[i2]];
    hasStereo = tier.channels[2] > 0;
    minHeight = Math.min(minHeight, tier.minHeight);
    minFramerate = Math.min(minFramerate, tier.minFramerate);
    minBitrate = Math.min(minBitrate, tier.minBitrate);
    const matchingVideoRanges = allowedVideoRanges.filter((range) => tier.videoRanges[range] > 0);
    if (matchingVideoRanges.length > 0) {
      hasCurrentVideoRange = true;
      videoRanges = matchingVideoRanges;
    }
  }
  minHeight = isFiniteNumber(minHeight) ? minHeight : 0;
  minFramerate = isFiniteNumber(minFramerate) ? minFramerate : 0;
  const maxHeight = Math.max(1080, minHeight);
  const maxFramerate = Math.max(30, minFramerate);
  minBitrate = isFiniteNumber(minBitrate) ? minBitrate : currentBw;
  currentBw = Math.max(minBitrate, currentBw);
  if (!hasCurrentVideoRange) {
    currentVideoRange = void 0;
    videoRanges = [];
  }
  const codecSet = codecSets.reduce((selected, candidate) => {
    const candidateTier = codecTiers[candidate];
    if (candidate === selected) {
      return selected;
    }
    if (candidateTier.minBitrate > currentBw) {
      logStartCodecCandidateIgnored(candidate, `min bitrate of ${candidateTier.minBitrate} > current estimate of ${currentBw}`);
      return selected;
    }
    if (!candidateTier.hasDefaultAudio) {
      logStartCodecCandidateIgnored(candidate, `no renditions with default or auto-select sound found`);
      return selected;
    }
    if (audioCodecPreference && candidate.indexOf(audioCodecPreference.substring(0, 4)) % 5 !== 0) {
      logStartCodecCandidateIgnored(candidate, `audio codec preference "${audioCodecPreference}" not found`);
      return selected;
    }
    if (channelsPreference && !preferStereo) {
      if (!candidateTier.channels[channelsPreference]) {
        logStartCodecCandidateIgnored(candidate, `no renditions with ${channelsPreference} channel sound found (channels options: ${Object.keys(candidateTier.channels)})`);
        return selected;
      }
    } else if ((!audioCodecPreference || preferStereo) && hasStereo && candidateTier.channels["2"] === 0) {
      logStartCodecCandidateIgnored(candidate, `no renditions with stereo sound found`);
      return selected;
    }
    if (candidateTier.minHeight > maxHeight) {
      logStartCodecCandidateIgnored(candidate, `min resolution of ${candidateTier.minHeight} > maximum of ${maxHeight}`);
      return selected;
    }
    if (candidateTier.minFramerate > maxFramerate) {
      logStartCodecCandidateIgnored(candidate, `min framerate of ${candidateTier.minFramerate} > maximum of ${maxFramerate}`);
      return selected;
    }
    if (!videoRanges.some((range) => candidateTier.videoRanges[range] > 0)) {
      logStartCodecCandidateIgnored(candidate, `no variants with VIDEO-RANGE of ${JSON.stringify(videoRanges)} found`);
      return selected;
    }
    if (candidateTier.maxScore < selectedScore) {
      logStartCodecCandidateIgnored(candidate, `max score of ${candidateTier.maxScore} < selected max of ${selectedScore}`);
      return selected;
    }
    if (selected && (codecsSetSelectionPreferenceValue(candidate) >= codecsSetSelectionPreferenceValue(selected) || candidateTier.fragmentError > codecTiers[selected].fragmentError)) {
      return selected;
    }
    selectedScore = candidateTier.maxScore;
    return candidate;
  }, void 0);
  return {
    codecSet,
    videoRanges,
    preferHDR,
    minFramerate,
    minBitrate
  };
}
function logStartCodecCandidateIgnored(codeSet, reason) {
  logger.log(`[abr] start candidates with "${codeSet}" ignored because ${reason}`);
}
function getAudioTracksByGroup(allAudioTracks) {
  return allAudioTracks.reduce((audioTracksByGroup, track) => {
    let trackGroup = audioTracksByGroup.groups[track.groupId];
    if (!trackGroup) {
      trackGroup = audioTracksByGroup.groups[track.groupId] = {
        tracks: [],
        channels: {
          2: 0
        },
        hasDefault: false,
        hasAutoSelect: false
      };
    }
    trackGroup.tracks.push(track);
    const channelsKey = track.channels || "2";
    trackGroup.channels[channelsKey] = (trackGroup.channels[channelsKey] || 0) + 1;
    trackGroup.hasDefault = trackGroup.hasDefault || track.default;
    trackGroup.hasAutoSelect = trackGroup.hasAutoSelect || track.autoselect;
    if (trackGroup.hasDefault) {
      audioTracksByGroup.hasDefaultAudio = true;
    }
    if (trackGroup.hasAutoSelect) {
      audioTracksByGroup.hasAutoSelectAudio = true;
    }
    return audioTracksByGroup;
  }, {
    hasDefaultAudio: false,
    hasAutoSelectAudio: false,
    groups: {}
  });
}
function getCodecTiers(levels, audioTracksByGroup, minAutoLevel, maxAutoLevel) {
  return levels.slice(minAutoLevel, maxAutoLevel + 1).reduce((tiers, level) => {
    if (!level.codecSet) {
      return tiers;
    }
    const audioGroups = level.audioGroups;
    let tier = tiers[level.codecSet];
    if (!tier) {
      tiers[level.codecSet] = tier = {
        minBitrate: Infinity,
        minHeight: Infinity,
        minFramerate: Infinity,
        maxScore: 0,
        videoRanges: {
          SDR: 0
        },
        channels: {
          "2": 0
        },
        hasDefaultAudio: !audioGroups,
        fragmentError: 0
      };
    }
    tier.minBitrate = Math.min(tier.minBitrate, level.bitrate);
    const lesserWidthOrHeight = Math.min(level.height, level.width);
    tier.minHeight = Math.min(tier.minHeight, lesserWidthOrHeight);
    tier.minFramerate = Math.min(tier.minFramerate, level.frameRate);
    tier.maxScore = Math.max(tier.maxScore, level.score);
    tier.fragmentError += level.fragmentError;
    tier.videoRanges[level.videoRange] = (tier.videoRanges[level.videoRange] || 0) + 1;
    if (audioGroups) {
      audioGroups.forEach((audioGroupId) => {
        if (!audioGroupId) {
          return;
        }
        const audioGroup = audioTracksByGroup.groups[audioGroupId];
        if (!audioGroup) {
          return;
        }
        tier.hasDefaultAudio = tier.hasDefaultAudio || audioTracksByGroup.hasDefaultAudio ? audioGroup.hasDefault : audioGroup.hasAutoSelect || !audioTracksByGroup.hasDefaultAudio && !audioTracksByGroup.hasAutoSelectAudio;
        Object.keys(audioGroup.channels).forEach((channels) => {
          tier.channels[channels] = (tier.channels[channels] || 0) + audioGroup.channels[channels];
        });
      });
    }
    return tiers;
  }, {});
}
function findMatchingOption(option, tracks, matchPredicate) {
  if ("attrs" in option) {
    const index = tracks.indexOf(option);
    if (index !== -1) {
      return index;
    }
  }
  for (let i2 = 0; i2 < tracks.length; i2++) {
    const track = tracks[i2];
    if (matchesOption(option, track, matchPredicate)) {
      return i2;
    }
  }
  return -1;
}
function matchesOption(option, track, matchPredicate) {
  const {
    groupId,
    name,
    lang,
    assocLang,
    characteristics,
    default: isDefault
  } = option;
  const forced = option.forced;
  return (groupId === void 0 || track.groupId === groupId) && (name === void 0 || track.name === name) && (lang === void 0 || track.lang === lang) && (lang === void 0 || track.assocLang === assocLang) && (isDefault === void 0 || track.default === isDefault) && (forced === void 0 || track.forced === forced) && (characteristics === void 0 || characteristicsMatch(characteristics, track.characteristics)) && (matchPredicate === void 0 || matchPredicate(option, track));
}
function characteristicsMatch(characteristicsA, characteristicsB = "") {
  const arrA = characteristicsA.split(",");
  const arrB = characteristicsB.split(",");
  return arrA.length === arrB.length && !arrA.some((el) => arrB.indexOf(el) === -1);
}
function audioMatchPredicate(option, track) {
  const {
    audioCodec,
    channels
  } = option;
  return (audioCodec === void 0 || (track.audioCodec || "").substring(0, 4) === audioCodec.substring(0, 4)) && (channels === void 0 || channels === (track.channels || "2"));
}
function findClosestLevelWithAudioGroup(option, levels, allAudioTracks, searchIndex, matchPredicate) {
  const currentLevel = levels[searchIndex];
  const variants = levels.reduce((variantMap, level, index) => {
    const uri = level.uri;
    const renditions2 = variantMap[uri] || (variantMap[uri] = []);
    renditions2.push(index);
    return variantMap;
  }, {});
  const renditions = variants[currentLevel.uri];
  if (renditions.length > 1) {
    searchIndex = Math.max.apply(Math, renditions);
  }
  const currentVideoRange = currentLevel.videoRange;
  const currentFrameRate = currentLevel.frameRate;
  const currentVideoCodec = currentLevel.codecSet.substring(0, 4);
  const matchingVideo = searchDownAndUpList(levels, searchIndex, (level) => {
    if (level.videoRange !== currentVideoRange || level.frameRate !== currentFrameRate || level.codecSet.substring(0, 4) !== currentVideoCodec) {
      return false;
    }
    const audioGroups = level.audioGroups;
    const tracks = allAudioTracks.filter((track) => !audioGroups || audioGroups.indexOf(track.groupId) !== -1);
    return findMatchingOption(option, tracks, matchPredicate) > -1;
  });
  if (matchingVideo > -1) {
    return matchingVideo;
  }
  return searchDownAndUpList(levels, searchIndex, (level) => {
    const audioGroups = level.audioGroups;
    const tracks = allAudioTracks.filter((track) => !audioGroups || audioGroups.indexOf(track.groupId) !== -1);
    return findMatchingOption(option, tracks, matchPredicate) > -1;
  });
}
function searchDownAndUpList(arr, searchIndex, predicate) {
  for (let i2 = searchIndex; i2; i2--) {
    if (predicate(arr[i2])) {
      return i2;
    }
  }
  for (let i2 = searchIndex + 1; i2 < arr.length; i2++) {
    if (predicate(arr[i2])) {
      return i2;
    }
  }
  return -1;
}
class AbrController {
  constructor(_hls) {
    this.hls = void 0;
    this.lastLevelLoadSec = 0;
    this.lastLoadedFragLevel = -1;
    this.firstSelection = -1;
    this._nextAutoLevel = -1;
    this.nextAutoLevelKey = "";
    this.audioTracksByGroup = null;
    this.codecTiers = null;
    this.timer = -1;
    this.fragCurrent = null;
    this.partCurrent = null;
    this.bitrateTestDelay = 0;
    this.bwEstimator = void 0;
    this._abandonRulesCheck = () => {
      const {
        fragCurrent: frag,
        partCurrent: part,
        hls
      } = this;
      const {
        autoLevelEnabled,
        media
      } = hls;
      if (!frag || !media) {
        return;
      }
      const now2 = performance.now();
      const stats = part ? part.stats : frag.stats;
      const duration = part ? part.duration : frag.duration;
      const timeLoading = now2 - stats.loading.start;
      const minAutoLevel = hls.minAutoLevel;
      if (stats.aborted || stats.loaded && stats.loaded === stats.total || frag.level <= minAutoLevel) {
        this.clearTimer();
        this._nextAutoLevel = -1;
        return;
      }
      if (!autoLevelEnabled || media.paused || !media.playbackRate || !media.readyState) {
        return;
      }
      const bufferInfo = hls.mainForwardBufferInfo;
      if (bufferInfo === null) {
        return;
      }
      const ttfbEstimate = this.bwEstimator.getEstimateTTFB();
      const playbackRate = Math.abs(media.playbackRate);
      if (timeLoading <= Math.max(ttfbEstimate, 1e3 * (duration / (playbackRate * 2)))) {
        return;
      }
      const bufferStarvationDelay = bufferInfo.len / playbackRate;
      const ttfb = stats.loading.first ? stats.loading.first - stats.loading.start : -1;
      const loadedFirstByte = stats.loaded && ttfb > -1;
      const bwEstimate = this.getBwEstimate();
      const levels = hls.levels;
      const level = levels[frag.level];
      const expectedLen = stats.total || Math.max(stats.loaded, Math.round(duration * level.averageBitrate / 8));
      let timeStreaming = loadedFirstByte ? timeLoading - ttfb : timeLoading;
      if (timeStreaming < 1 && loadedFirstByte) {
        timeStreaming = Math.min(timeLoading, stats.loaded * 8 / bwEstimate);
      }
      const loadRate = loadedFirstByte ? stats.loaded * 1e3 / timeStreaming : 0;
      const fragLoadedDelay = loadRate ? (expectedLen - stats.loaded) / loadRate : expectedLen * 8 / bwEstimate + ttfbEstimate / 1e3;
      if (fragLoadedDelay <= bufferStarvationDelay) {
        return;
      }
      const bwe = loadRate ? loadRate * 8 : bwEstimate;
      let fragLevelNextLoadedDelay = Number.POSITIVE_INFINITY;
      let nextLoadLevel;
      for (nextLoadLevel = frag.level - 1; nextLoadLevel > minAutoLevel; nextLoadLevel--) {
        const levelNextBitrate = levels[nextLoadLevel].maxBitrate;
        fragLevelNextLoadedDelay = this.getTimeToLoadFrag(ttfbEstimate / 1e3, bwe, duration * levelNextBitrate, !levels[nextLoadLevel].details);
        if (fragLevelNextLoadedDelay < bufferStarvationDelay) {
          break;
        }
      }
      if (fragLevelNextLoadedDelay >= fragLoadedDelay) {
        return;
      }
      if (fragLevelNextLoadedDelay > duration * 10) {
        return;
      }
      hls.nextLoadLevel = hls.nextAutoLevel = nextLoadLevel;
      if (loadedFirstByte) {
        this.bwEstimator.sample(timeLoading - Math.min(ttfbEstimate, ttfb), stats.loaded);
      } else {
        this.bwEstimator.sampleTTFB(timeLoading);
      }
      const nextLoadLevelBitrate = levels[nextLoadLevel].maxBitrate;
      if (this.getBwEstimate() * this.hls.config.abrBandWidthUpFactor > nextLoadLevelBitrate) {
        this.resetEstimator(nextLoadLevelBitrate);
      }
      this.clearTimer();
      logger.warn(`[abr] Fragment ${frag.sn}${part ? " part " + part.index : ""} of level ${frag.level} is loading too slowly;
      Time to underbuffer: ${bufferStarvationDelay.toFixed(3)} s
      Estimated load time for current fragment: ${fragLoadedDelay.toFixed(3)} s
      Estimated load time for down switch fragment: ${fragLevelNextLoadedDelay.toFixed(3)} s
      TTFB estimate: ${ttfb | 0} ms
      Current BW estimate: ${isFiniteNumber(bwEstimate) ? bwEstimate | 0 : "Unknown"} bps
      New BW estimate: ${this.getBwEstimate() | 0} bps
      Switching to level ${nextLoadLevel} @ ${nextLoadLevelBitrate | 0} bps`);
      hls.trigger(Events.FRAG_LOAD_EMERGENCY_ABORTED, {
        frag,
        part,
        stats
      });
    };
    this.hls = _hls;
    this.bwEstimator = this.initEstimator();
    this.registerListeners();
  }
  resetEstimator(abrEwmaDefaultEstimate) {
    if (abrEwmaDefaultEstimate) {
      logger.log(`setting initial bwe to ${abrEwmaDefaultEstimate}`);
      this.hls.config.abrEwmaDefaultEstimate = abrEwmaDefaultEstimate;
    }
    this.firstSelection = -1;
    this.bwEstimator = this.initEstimator();
  }
  initEstimator() {
    const config = this.hls.config;
    return new EwmaBandWidthEstimator(config.abrEwmaSlowVoD, config.abrEwmaFastVoD, config.abrEwmaDefaultEstimate);
  }
  registerListeners() {
    const {
      hls
    } = this;
    hls.on(Events.MANIFEST_LOADING, this.onManifestLoading, this);
    hls.on(Events.FRAG_LOADING, this.onFragLoading, this);
    hls.on(Events.FRAG_LOADED, this.onFragLoaded, this);
    hls.on(Events.FRAG_BUFFERED, this.onFragBuffered, this);
    hls.on(Events.LEVEL_SWITCHING, this.onLevelSwitching, this);
    hls.on(Events.LEVEL_LOADED, this.onLevelLoaded, this);
    hls.on(Events.LEVELS_UPDATED, this.onLevelsUpdated, this);
    hls.on(Events.MAX_AUTO_LEVEL_UPDATED, this.onMaxAutoLevelUpdated, this);
    hls.on(Events.ERROR, this.onError, this);
  }
  unregisterListeners() {
    const {
      hls
    } = this;
    if (!hls) {
      return;
    }
    hls.off(Events.MANIFEST_LOADING, this.onManifestLoading, this);
    hls.off(Events.FRAG_LOADING, this.onFragLoading, this);
    hls.off(Events.FRAG_LOADED, this.onFragLoaded, this);
    hls.off(Events.FRAG_BUFFERED, this.onFragBuffered, this);
    hls.off(Events.LEVEL_SWITCHING, this.onLevelSwitching, this);
    hls.off(Events.LEVEL_LOADED, this.onLevelLoaded, this);
    hls.off(Events.LEVELS_UPDATED, this.onLevelsUpdated, this);
    hls.off(Events.MAX_AUTO_LEVEL_UPDATED, this.onMaxAutoLevelUpdated, this);
    hls.off(Events.ERROR, this.onError, this);
  }
  destroy() {
    this.unregisterListeners();
    this.clearTimer();
    this.hls = this._abandonRulesCheck = null;
    this.fragCurrent = this.partCurrent = null;
  }
  onManifestLoading(event, data) {
    this.lastLoadedFragLevel = -1;
    this.firstSelection = -1;
    this.lastLevelLoadSec = 0;
    this.fragCurrent = this.partCurrent = null;
    this.onLevelsUpdated();
    this.clearTimer();
  }
  onLevelsUpdated() {
    if (this.lastLoadedFragLevel > -1 && this.fragCurrent) {
      this.lastLoadedFragLevel = this.fragCurrent.level;
    }
    this._nextAutoLevel = -1;
    this.onMaxAutoLevelUpdated();
    this.codecTiers = null;
    this.audioTracksByGroup = null;
  }
  onMaxAutoLevelUpdated() {
    this.firstSelection = -1;
    this.nextAutoLevelKey = "";
  }
  onFragLoading(event, data) {
    const frag = data.frag;
    if (this.ignoreFragment(frag)) {
      return;
    }
    if (!frag.bitrateTest) {
      var _data$part;
      this.fragCurrent = frag;
      this.partCurrent = (_data$part = data.part) != null ? _data$part : null;
    }
    this.clearTimer();
    this.timer = self.setInterval(this._abandonRulesCheck, 100);
  }
  onLevelSwitching(event, data) {
    this.clearTimer();
  }
  onError(event, data) {
    if (data.fatal) {
      return;
    }
    switch (data.details) {
      case ErrorDetails.BUFFER_ADD_CODEC_ERROR:
      case ErrorDetails.BUFFER_APPEND_ERROR:
        this.lastLoadedFragLevel = -1;
        this.firstSelection = -1;
        break;
      case ErrorDetails.FRAG_LOAD_TIMEOUT: {
        const frag = data.frag;
        const {
          fragCurrent,
          partCurrent: part
        } = this;
        if (frag && fragCurrent && frag.sn === fragCurrent.sn && frag.level === fragCurrent.level) {
          const now2 = performance.now();
          const stats = part ? part.stats : frag.stats;
          const timeLoading = now2 - stats.loading.start;
          const ttfb = stats.loading.first ? stats.loading.first - stats.loading.start : -1;
          const loadedFirstByte = stats.loaded && ttfb > -1;
          if (loadedFirstByte) {
            const ttfbEstimate = this.bwEstimator.getEstimateTTFB();
            this.bwEstimator.sample(timeLoading - Math.min(ttfbEstimate, ttfb), stats.loaded);
          } else {
            this.bwEstimator.sampleTTFB(timeLoading);
          }
        }
        break;
      }
    }
  }
  getTimeToLoadFrag(timeToFirstByteSec, bandwidth, fragSizeBits, isSwitch) {
    const fragLoadSec = timeToFirstByteSec + fragSizeBits / bandwidth;
    const playlistLoadSec = isSwitch ? this.lastLevelLoadSec : 0;
    return fragLoadSec + playlistLoadSec;
  }
  onLevelLoaded(event, data) {
    const config = this.hls.config;
    const {
      loading
    } = data.stats;
    const timeLoadingMs = loading.end - loading.start;
    if (isFiniteNumber(timeLoadingMs)) {
      this.lastLevelLoadSec = timeLoadingMs / 1e3;
    }
    if (data.details.live) {
      this.bwEstimator.update(config.abrEwmaSlowLive, config.abrEwmaFastLive);
    } else {
      this.bwEstimator.update(config.abrEwmaSlowVoD, config.abrEwmaFastVoD);
    }
  }
  onFragLoaded(event, {
    frag,
    part
  }) {
    const stats = part ? part.stats : frag.stats;
    if (frag.type === PlaylistLevelType.MAIN) {
      this.bwEstimator.sampleTTFB(stats.loading.first - stats.loading.start);
    }
    if (this.ignoreFragment(frag)) {
      return;
    }
    this.clearTimer();
    if (frag.level === this._nextAutoLevel) {
      this._nextAutoLevel = -1;
    }
    this.firstSelection = -1;
    if (this.hls.config.abrMaxWithRealBitrate) {
      const duration = part ? part.duration : frag.duration;
      const level = this.hls.levels[frag.level];
      const loadedBytes = (level.loaded ? level.loaded.bytes : 0) + stats.loaded;
      const loadedDuration = (level.loaded ? level.loaded.duration : 0) + duration;
      level.loaded = {
        bytes: loadedBytes,
        duration: loadedDuration
      };
      level.realBitrate = Math.round(8 * loadedBytes / loadedDuration);
    }
    if (frag.bitrateTest) {
      const fragBufferedData = {
        stats,
        frag,
        part,
        id: frag.type
      };
      this.onFragBuffered(Events.FRAG_BUFFERED, fragBufferedData);
      frag.bitrateTest = false;
    } else {
      this.lastLoadedFragLevel = frag.level;
    }
  }
  onFragBuffered(event, data) {
    const {
      frag,
      part
    } = data;
    const stats = part != null && part.stats.loaded ? part.stats : frag.stats;
    if (stats.aborted) {
      return;
    }
    if (this.ignoreFragment(frag)) {
      return;
    }
    const processingMs = stats.parsing.end - stats.loading.start - Math.min(stats.loading.first - stats.loading.start, this.bwEstimator.getEstimateTTFB());
    this.bwEstimator.sample(processingMs, stats.loaded);
    stats.bwEstimate = this.getBwEstimate();
    if (frag.bitrateTest) {
      this.bitrateTestDelay = processingMs / 1e3;
    } else {
      this.bitrateTestDelay = 0;
    }
  }
  ignoreFragment(frag) {
    return frag.type !== PlaylistLevelType.MAIN || frag.sn === "initSegment";
  }
  clearTimer() {
    if (this.timer > -1) {
      self.clearInterval(this.timer);
      this.timer = -1;
    }
  }
  get firstAutoLevel() {
    const {
      maxAutoLevel,
      minAutoLevel
    } = this.hls;
    const bwEstimate = this.getBwEstimate();
    const maxStartDelay = this.hls.config.maxStarvationDelay;
    const abrAutoLevel = this.findBestLevel(bwEstimate, minAutoLevel, maxAutoLevel, 0, maxStartDelay, 1, 1);
    if (abrAutoLevel > -1) {
      return abrAutoLevel;
    }
    const firstLevel = this.hls.firstLevel;
    const clamped = Math.min(Math.max(firstLevel, minAutoLevel), maxAutoLevel);
    logger.warn(`[abr] Could not find best starting auto level. Defaulting to first in playlist ${firstLevel} clamped to ${clamped}`);
    return clamped;
  }
  get forcedAutoLevel() {
    if (this.nextAutoLevelKey) {
      return -1;
    }
    return this._nextAutoLevel;
  }
  // return next auto level
  get nextAutoLevel() {
    const forcedAutoLevel = this.forcedAutoLevel;
    const bwEstimator = this.bwEstimator;
    const useEstimate = bwEstimator.canEstimate();
    const loadedFirstFrag = this.lastLoadedFragLevel > -1;
    if (forcedAutoLevel !== -1 && (!useEstimate || !loadedFirstFrag || this.nextAutoLevelKey === this.getAutoLevelKey())) {
      return forcedAutoLevel;
    }
    const nextABRAutoLevel = useEstimate && loadedFirstFrag ? this.getNextABRAutoLevel() : this.firstAutoLevel;
    if (forcedAutoLevel !== -1) {
      const levels = this.hls.levels;
      if (levels.length > Math.max(forcedAutoLevel, nextABRAutoLevel) && levels[forcedAutoLevel].loadError <= levels[nextABRAutoLevel].loadError) {
        return forcedAutoLevel;
      }
    }
    this._nextAutoLevel = nextABRAutoLevel;
    this.nextAutoLevelKey = this.getAutoLevelKey();
    return nextABRAutoLevel;
  }
  getAutoLevelKey() {
    return `${this.getBwEstimate()}_${this.getStarvationDelay().toFixed(2)}`;
  }
  getNextABRAutoLevel() {
    const {
      fragCurrent,
      partCurrent,
      hls
    } = this;
    const {
      maxAutoLevel,
      config,
      minAutoLevel
    } = hls;
    const currentFragDuration = partCurrent ? partCurrent.duration : fragCurrent ? fragCurrent.duration : 0;
    const avgbw = this.getBwEstimate();
    const bufferStarvationDelay = this.getStarvationDelay();
    let bwFactor = config.abrBandWidthFactor;
    let bwUpFactor = config.abrBandWidthUpFactor;
    if (bufferStarvationDelay) {
      const _bestLevel = this.findBestLevel(avgbw, minAutoLevel, maxAutoLevel, bufferStarvationDelay, 0, bwFactor, bwUpFactor);
      if (_bestLevel >= 0) {
        return _bestLevel;
      }
    }
    let maxStarvationDelay = currentFragDuration ? Math.min(currentFragDuration, config.maxStarvationDelay) : config.maxStarvationDelay;
    if (!bufferStarvationDelay) {
      const bitrateTestDelay = this.bitrateTestDelay;
      if (bitrateTestDelay) {
        const maxLoadingDelay = currentFragDuration ? Math.min(currentFragDuration, config.maxLoadingDelay) : config.maxLoadingDelay;
        maxStarvationDelay = maxLoadingDelay - bitrateTestDelay;
        logger.info(`[abr] bitrate test took ${Math.round(1e3 * bitrateTestDelay)}ms, set first fragment max fetchDuration to ${Math.round(1e3 * maxStarvationDelay)} ms`);
        bwFactor = bwUpFactor = 1;
      }
    }
    const bestLevel = this.findBestLevel(avgbw, minAutoLevel, maxAutoLevel, bufferStarvationDelay, maxStarvationDelay, bwFactor, bwUpFactor);
    logger.info(`[abr] ${bufferStarvationDelay ? "rebuffering expected" : "buffer is empty"}, optimal quality level ${bestLevel}`);
    if (bestLevel > -1) {
      return bestLevel;
    }
    const minLevel = hls.levels[minAutoLevel];
    const autoLevel = hls.levels[hls.loadLevel];
    if ((minLevel == null ? void 0 : minLevel.bitrate) < (autoLevel == null ? void 0 : autoLevel.bitrate)) {
      return minAutoLevel;
    }
    return hls.loadLevel;
  }
  getStarvationDelay() {
    const hls = this.hls;
    const media = hls.media;
    if (!media) {
      return Infinity;
    }
    const playbackRate = media && media.playbackRate !== 0 ? Math.abs(media.playbackRate) : 1;
    const bufferInfo = hls.mainForwardBufferInfo;
    return (bufferInfo ? bufferInfo.len : 0) / playbackRate;
  }
  getBwEstimate() {
    return this.bwEstimator.canEstimate() ? this.bwEstimator.getEstimate() : this.hls.config.abrEwmaDefaultEstimate;
  }
  findBestLevel(currentBw, minAutoLevel, maxAutoLevel, bufferStarvationDelay, maxStarvationDelay, bwFactor, bwUpFactor) {
    var _level$details;
    const maxFetchDuration = bufferStarvationDelay + maxStarvationDelay;
    const lastLoadedFragLevel = this.lastLoadedFragLevel;
    const selectionBaseLevel = lastLoadedFragLevel === -1 ? this.hls.firstLevel : lastLoadedFragLevel;
    const {
      fragCurrent,
      partCurrent
    } = this;
    const {
      levels,
      allAudioTracks,
      loadLevel,
      config
    } = this.hls;
    if (levels.length === 1) {
      return 0;
    }
    const level = levels[selectionBaseLevel];
    const live = !!(level != null && (_level$details = level.details) != null && _level$details.live);
    const firstSelection = loadLevel === -1 || lastLoadedFragLevel === -1;
    let currentCodecSet;
    let currentVideoRange = "SDR";
    let currentFrameRate = (level == null ? void 0 : level.frameRate) || 0;
    const {
      audioPreference,
      videoPreference
    } = config;
    const audioTracksByGroup = this.audioTracksByGroup || (this.audioTracksByGroup = getAudioTracksByGroup(allAudioTracks));
    if (firstSelection) {
      if (this.firstSelection !== -1) {
        return this.firstSelection;
      }
      const codecTiers = this.codecTiers || (this.codecTiers = getCodecTiers(levels, audioTracksByGroup, minAutoLevel, maxAutoLevel));
      const startTier = getStartCodecTier(codecTiers, currentVideoRange, currentBw, audioPreference, videoPreference);
      const {
        codecSet,
        videoRanges,
        minFramerate,
        minBitrate,
        preferHDR
      } = startTier;
      currentCodecSet = codecSet;
      currentVideoRange = preferHDR ? videoRanges[videoRanges.length - 1] : videoRanges[0];
      currentFrameRate = minFramerate;
      currentBw = Math.max(currentBw, minBitrate);
      logger.log(`[abr] picked start tier ${JSON.stringify(startTier)}`);
    } else {
      currentCodecSet = level == null ? void 0 : level.codecSet;
      currentVideoRange = level == null ? void 0 : level.videoRange;
    }
    const currentFragDuration = partCurrent ? partCurrent.duration : fragCurrent ? fragCurrent.duration : 0;
    const ttfbEstimateSec = this.bwEstimator.getEstimateTTFB() / 1e3;
    const levelsSkipped = [];
    for (let i2 = maxAutoLevel; i2 >= minAutoLevel; i2--) {
      var _levelInfo$supportedR;
      const levelInfo = levels[i2];
      const upSwitch = i2 > selectionBaseLevel;
      if (!levelInfo) {
        continue;
      }
      if (config.useMediaCapabilities && !levelInfo.supportedResult && !levelInfo.supportedPromise) {
        const mediaCapabilities = navigator.mediaCapabilities;
        if (typeof (mediaCapabilities == null ? void 0 : mediaCapabilities.decodingInfo) === "function" && requiresMediaCapabilitiesDecodingInfo(levelInfo, audioTracksByGroup, currentVideoRange, currentFrameRate, currentBw, audioPreference)) {
          levelInfo.supportedPromise = getMediaDecodingInfoPromise(levelInfo, audioTracksByGroup, mediaCapabilities);
          levelInfo.supportedPromise.then((decodingInfo) => {
            if (!this.hls) {
              return;
            }
            levelInfo.supportedResult = decodingInfo;
            const levels2 = this.hls.levels;
            const index = levels2.indexOf(levelInfo);
            if (decodingInfo.error) {
              logger.warn(`[abr] MediaCapabilities decodingInfo error: "${decodingInfo.error}" for level ${index} ${JSON.stringify(decodingInfo)}`);
            } else if (!decodingInfo.supported) {
              logger.warn(`[abr] Unsupported MediaCapabilities decodingInfo result for level ${index} ${JSON.stringify(decodingInfo)}`);
              if (index > -1 && levels2.length > 1) {
                logger.log(`[abr] Removing unsupported level ${index}`);
                this.hls.removeLevel(index);
              }
            }
          });
        } else {
          levelInfo.supportedResult = SUPPORTED_INFO_DEFAULT;
        }
      }
      if (currentCodecSet && levelInfo.codecSet !== currentCodecSet || currentVideoRange && levelInfo.videoRange !== currentVideoRange || upSwitch && currentFrameRate > levelInfo.frameRate || !upSwitch && currentFrameRate > 0 && currentFrameRate < levelInfo.frameRate || levelInfo.supportedResult && !((_levelInfo$supportedR = levelInfo.supportedResult.decodingInfoResults) != null && _levelInfo$supportedR[0].smooth)) {
        levelsSkipped.push(i2);
        continue;
      }
      const levelDetails = levelInfo.details;
      const avgDuration = (partCurrent ? levelDetails == null ? void 0 : levelDetails.partTarget : levelDetails == null ? void 0 : levelDetails.averagetargetduration) || currentFragDuration;
      let adjustedbw;
      if (!upSwitch) {
        adjustedbw = bwFactor * currentBw;
      } else {
        adjustedbw = bwUpFactor * currentBw;
      }
      const bitrate = currentFragDuration && bufferStarvationDelay >= currentFragDuration * 2 && maxStarvationDelay === 0 ? levels[i2].averageBitrate : levels[i2].maxBitrate;
      const fetchDuration = this.getTimeToLoadFrag(ttfbEstimateSec, adjustedbw, bitrate * avgDuration, levelDetails === void 0);
      const canSwitchWithinTolerance = (
        // if adjusted bw is greater than level bitrate AND
        adjustedbw >= bitrate && // no level change, or new level has no error history
        (i2 === lastLoadedFragLevel || levelInfo.loadError === 0 && levelInfo.fragmentError === 0) && // fragment fetchDuration unknown OR live stream OR fragment fetchDuration less than max allowed fetch duration, then this level matches
        // we don't account for max Fetch Duration for live streams, this is to avoid switching down when near the edge of live sliding window ...
        // special case to support startLevel = -1 (bitrateTest) on live streams : in that case we should not exit loop so that findBestLevel will return -1
        (fetchDuration <= ttfbEstimateSec || !isFiniteNumber(fetchDuration) || live && !this.bitrateTestDelay || fetchDuration < maxFetchDuration)
      );
      if (canSwitchWithinTolerance) {
        const forcedAutoLevel = this.forcedAutoLevel;
        if (i2 !== loadLevel && (forcedAutoLevel === -1 || forcedAutoLevel !== loadLevel)) {
          if (levelsSkipped.length) {
            logger.trace(`[abr] Skipped level(s) ${levelsSkipped.join(",")} of ${maxAutoLevel} max with CODECS and VIDEO-RANGE:"${levels[levelsSkipped[0]].codecs}" ${levels[levelsSkipped[0]].videoRange}; not compatible with "${level.codecs}" ${currentVideoRange}`);
          }
          logger.info(`[abr] switch candidate:${selectionBaseLevel}->${i2} adjustedbw(${Math.round(adjustedbw)})-bitrate=${Math.round(adjustedbw - bitrate)} ttfb:${ttfbEstimateSec.toFixed(1)} avgDuration:${avgDuration.toFixed(1)} maxFetchDuration:${maxFetchDuration.toFixed(1)} fetchDuration:${fetchDuration.toFixed(1)} firstSelection:${firstSelection} codecSet:${currentCodecSet} videoRange:${currentVideoRange} hls.loadLevel:${loadLevel}`);
        }
        if (firstSelection) {
          this.firstSelection = i2;
        }
        return i2;
      }
    }
    return -1;
  }
  set nextAutoLevel(nextLevel) {
    const {
      maxAutoLevel,
      minAutoLevel
    } = this.hls;
    const value = Math.min(Math.max(nextLevel, minAutoLevel), maxAutoLevel);
    if (this._nextAutoLevel !== value) {
      this.nextAutoLevelKey = "";
      this._nextAutoLevel = value;
    }
  }
}
class TaskLoop {
  constructor() {
    this._boundTick = void 0;
    this._tickTimer = null;
    this._tickInterval = null;
    this._tickCallCount = 0;
    this._boundTick = this.tick.bind(this);
  }
  destroy() {
    this.onHandlerDestroying();
    this.onHandlerDestroyed();
  }
  onHandlerDestroying() {
    this.clearNextTick();
    this.clearInterval();
  }
  onHandlerDestroyed() {
  }
  hasInterval() {
    return !!this._tickInterval;
  }
  hasNextTick() {
    return !!this._tickTimer;
  }
  /**
   * @param millis - Interval time (ms)
   * @eturns True when interval has been scheduled, false when already scheduled (no effect)
   */
  setInterval(millis) {
    if (!this._tickInterval) {
      this._tickCallCount = 0;
      this._tickInterval = self.setInterval(this._boundTick, millis);
      return true;
    }
    return false;
  }
  /**
   * @returns True when interval was cleared, false when none was set (no effect)
   */
  clearInterval() {
    if (this._tickInterval) {
      self.clearInterval(this._tickInterval);
      this._tickInterval = null;
      return true;
    }
    return false;
  }
  /**
   * @returns True when timeout was cleared, false when none was set (no effect)
   */
  clearNextTick() {
    if (this._tickTimer) {
      self.clearTimeout(this._tickTimer);
      this._tickTimer = null;
      return true;
    }
    return false;
  }
  /**
   * Will call the subclass doTick implementation in this main loop tick
   * or in the next one (via setTimeout(,0)) in case it has already been called
   * in this tick (in case this is a re-entrant call).
   */
  tick() {
    this._tickCallCount++;
    if (this._tickCallCount === 1) {
      this.doTick();
      if (this._tickCallCount > 1) {
        this.tickImmediate();
      }
      this._tickCallCount = 0;
    }
  }
  tickImmediate() {
    this.clearNextTick();
    this._tickTimer = self.setTimeout(this._boundTick, 0);
  }
  /**
   * For subclass to implement task logic
   * @abstract
   */
  doTick() {
  }
}
var FragmentState = {
  NOT_LOADED: "NOT_LOADED",
  APPENDING: "APPENDING",
  PARTIAL: "PARTIAL",
  OK: "OK"
};
const noopBuffered = {
  length: 0,
  start: () => 0,
  end: () => 0
};
class BufferHelper {
  /**
   * Return true if `media`'s buffered include `position`
   */
  static isBuffered(media, position) {
    try {
      if (media) {
        const buffered = BufferHelper.getBuffered(media);
        for (let i2 = 0; i2 < buffered.length; i2++) {
          if (position >= buffered.start(i2) && position <= buffered.end(i2)) {
            return true;
          }
        }
      }
    } catch (error) {
    }
    return false;
  }
  static bufferInfo(media, pos, maxHoleDuration) {
    try {
      if (media) {
        const vbuffered = BufferHelper.getBuffered(media);
        const buffered = [];
        let i2;
        for (i2 = 0; i2 < vbuffered.length; i2++) {
          buffered.push({
            start: vbuffered.start(i2),
            end: vbuffered.end(i2)
          });
        }
        return this.bufferedInfo(buffered, pos, maxHoleDuration);
      }
    } catch (error) {
    }
    return {
      len: 0,
      start: pos,
      end: pos,
      nextStart: void 0
    };
  }
  static bufferedInfo(buffered, pos, maxHoleDuration) {
    pos = Math.max(0, pos);
    buffered.sort(function(a, b) {
      const diff = a.start - b.start;
      if (diff) {
        return diff;
      } else {
        return b.end - a.end;
      }
    });
    let buffered2 = [];
    if (maxHoleDuration) {
      for (let i2 = 0; i2 < buffered.length; i2++) {
        const buf2len = buffered2.length;
        if (buf2len) {
          const buf2end = buffered2[buf2len - 1].end;
          if (buffered[i2].start - buf2end < maxHoleDuration) {
            if (buffered[i2].end > buf2end) {
              buffered2[buf2len - 1].end = buffered[i2].end;
            }
          } else {
            buffered2.push(buffered[i2]);
          }
        } else {
          buffered2.push(buffered[i2]);
        }
      }
    } else {
      buffered2 = buffered;
    }
    let bufferLen = 0;
    let bufferStartNext;
    let bufferStart = pos;
    let bufferEnd = pos;
    for (let i2 = 0; i2 < buffered2.length; i2++) {
      const start = buffered2[i2].start;
      const end = buffered2[i2].end;
      if (pos + maxHoleDuration >= start && pos < end) {
        bufferStart = start;
        bufferEnd = end;
        bufferLen = bufferEnd - pos;
      } else if (pos + maxHoleDuration < start) {
        bufferStartNext = start;
        break;
      }
    }
    return {
      len: bufferLen,
      start: bufferStart || 0,
      end: bufferEnd || 0,
      nextStart: bufferStartNext
    };
  }
  /**
   * Safe method to get buffered property.
   * SourceBuffer.buffered may throw if SourceBuffer is removed from it's MediaSource
   */
  static getBuffered(media) {
    try {
      return media.buffered;
    } catch (e2) {
      logger.log("failed to get media.buffered", e2);
      return noopBuffered;
    }
  }
}
class ChunkMetadata {
  constructor(level, sn, id, size = 0, part = -1, partial = false) {
    this.level = void 0;
    this.sn = void 0;
    this.part = void 0;
    this.id = void 0;
    this.size = void 0;
    this.partial = void 0;
    this.transmuxing = getNewPerformanceTiming();
    this.buffering = {
      audio: getNewPerformanceTiming(),
      video: getNewPerformanceTiming(),
      audiovideo: getNewPerformanceTiming()
    };
    this.level = level;
    this.sn = sn;
    this.id = id;
    this.size = size;
    this.part = part;
    this.partial = partial;
  }
}
function getNewPerformanceTiming() {
  return {
    start: 0,
    executeStart: 0,
    executeEnd: 0,
    end: 0
  };
}
function findFirstFragWithCC(fragments, cc) {
  for (let i2 = 0, len = fragments.length; i2 < len; i2++) {
    var _fragments$i;
    if (((_fragments$i = fragments[i2]) == null ? void 0 : _fragments$i.cc) === cc) {
      return fragments[i2];
    }
  }
  return null;
}
function shouldAlignOnDiscontinuities(lastFrag, switchDetails, details) {
  if (switchDetails) {
    if (details.endCC > details.startCC || lastFrag && lastFrag.cc < details.startCC) {
      return true;
    }
  }
  return false;
}
function findDiscontinuousReferenceFrag(prevDetails, curDetails) {
  const prevFrags = prevDetails.fragments;
  const curFrags = curDetails.fragments;
  if (!curFrags.length || !prevFrags.length) {
    logger.log("No fragments to align");
    return;
  }
  const prevStartFrag = findFirstFragWithCC(prevFrags, curFrags[0].cc);
  if (!prevStartFrag || prevStartFrag && !prevStartFrag.startPTS) {
    logger.log("No frag in previous level to align on");
    return;
  }
  return prevStartFrag;
}
function adjustFragmentStart(frag, sliding) {
  if (frag) {
    const start = frag.start + sliding;
    frag.start = frag.startPTS = start;
    frag.endPTS = start + frag.duration;
  }
}
function adjustSlidingStart(sliding, details) {
  const fragments = details.fragments;
  for (let i2 = 0, len = fragments.length; i2 < len; i2++) {
    adjustFragmentStart(fragments[i2], sliding);
  }
  if (details.fragmentHint) {
    adjustFragmentStart(details.fragmentHint, sliding);
  }
  details.alignedSliding = true;
}
function alignStream(lastFrag, switchDetails, details) {
  if (!switchDetails) {
    return;
  }
  alignDiscontinuities(lastFrag, details, switchDetails);
  if (!details.alignedSliding && switchDetails) {
    alignMediaPlaylistByPDT(details, switchDetails);
  }
  if (!details.alignedSliding && switchDetails && !details.skippedSegments) {
    adjustSliding(switchDetails, details);
  }
}
function alignDiscontinuities(lastFrag, details, switchDetails) {
  if (shouldAlignOnDiscontinuities(lastFrag, switchDetails, details)) {
    const referenceFrag = findDiscontinuousReferenceFrag(switchDetails, details);
    if (referenceFrag && isFiniteNumber(referenceFrag.start)) {
      logger.log(`Adjusting PTS using last level due to CC increase within current level ${details.url}`);
      adjustSlidingStart(referenceFrag.start, details);
    }
  }
}
function alignMediaPlaylistByPDT(details, refDetails) {
  if (!details.hasProgramDateTime || !refDetails.hasProgramDateTime) {
    return;
  }
  const fragments = details.fragments;
  const refFragments = refDetails.fragments;
  if (!fragments.length || !refFragments.length) {
    return;
  }
  let refFrag;
  let frag;
  const targetCC = Math.min(refDetails.endCC, details.endCC);
  if (refDetails.startCC < targetCC && details.startCC < targetCC) {
    refFrag = findFirstFragWithCC(refFragments, targetCC);
    frag = findFirstFragWithCC(fragments, targetCC);
  }
  if (!refFrag || !frag) {
    refFrag = refFragments[Math.floor(refFragments.length / 2)];
    frag = findFirstFragWithCC(fragments, refFrag.cc) || fragments[Math.floor(fragments.length / 2)];
  }
  const refPDT = refFrag.programDateTime;
  const targetPDT = frag.programDateTime;
  if (!refPDT || !targetPDT) {
    return;
  }
  const delta = (targetPDT - refPDT) / 1e3 - (frag.start - refFrag.start);
  adjustSlidingStart(delta, details);
}
const MIN_CHUNK_SIZE = Math.pow(2, 17);
class FragmentLoader {
  constructor(config) {
    this.config = void 0;
    this.loader = null;
    this.partLoadTimeout = -1;
    this.config = config;
  }
  destroy() {
    if (this.loader) {
      this.loader.destroy();
      this.loader = null;
    }
  }
  abort() {
    if (this.loader) {
      this.loader.abort();
    }
  }
  load(frag, onProgress) {
    const url = frag.url;
    if (!url) {
      return Promise.reject(new LoadError({
        type: ErrorTypes.NETWORK_ERROR,
        details: ErrorDetails.FRAG_LOAD_ERROR,
        fatal: false,
        frag,
        error: new Error(`Fragment does not have a ${url ? "part list" : "url"}`),
        networkDetails: null
      }));
    }
    this.abort();
    const config = this.config;
    const FragmentILoader = config.fLoader;
    const DefaultILoader = config.loader;
    return new Promise((resolve, reject) => {
      if (this.loader) {
        this.loader.destroy();
      }
      if (frag.gap) {
        if (frag.tagList.some((tags) => tags[0] === "GAP")) {
          reject(createGapLoadError(frag));
          return;
        } else {
          frag.gap = false;
        }
      }
      const loader = this.loader = frag.loader = FragmentILoader ? new FragmentILoader(config) : new DefaultILoader(config);
      const loaderContext = createLoaderContext(frag);
      const loadPolicy = getLoaderConfigWithoutReties(config.fragLoadPolicy.default);
      const loaderConfig = {
        loadPolicy,
        timeout: loadPolicy.maxLoadTimeMs,
        maxRetry: 0,
        retryDelay: 0,
        maxRetryDelay: 0,
        highWaterMark: frag.sn === "initSegment" ? Infinity : MIN_CHUNK_SIZE
      };
      frag.stats = loader.stats;
      loader.load(loaderContext, loaderConfig, {
        onSuccess: (response, stats, context, networkDetails) => {
          this.resetLoader(frag, loader);
          let payload = response.data;
          if (context.resetIV && frag.decryptdata) {
            frag.decryptdata.iv = new Uint8Array(payload.slice(0, 16));
            payload = payload.slice(16);
          }
          resolve({
            frag,
            part: null,
            payload,
            networkDetails
          });
        },
        onError: (response, context, networkDetails, stats) => {
          this.resetLoader(frag, loader);
          reject(new LoadError({
            type: ErrorTypes.NETWORK_ERROR,
            details: ErrorDetails.FRAG_LOAD_ERROR,
            fatal: false,
            frag,
            response: _objectSpread2({
              url,
              data: void 0
            }, response),
            error: new Error(`HTTP Error ${response.code} ${response.text}`),
            networkDetails,
            stats
          }));
        },
        onAbort: (stats, context, networkDetails) => {
          this.resetLoader(frag, loader);
          reject(new LoadError({
            type: ErrorTypes.NETWORK_ERROR,
            details: ErrorDetails.INTERNAL_ABORTED,
            fatal: false,
            frag,
            error: new Error("Aborted"),
            networkDetails,
            stats
          }));
        },
        onTimeout: (stats, context, networkDetails) => {
          this.resetLoader(frag, loader);
          reject(new LoadError({
            type: ErrorTypes.NETWORK_ERROR,
            details: ErrorDetails.FRAG_LOAD_TIMEOUT,
            fatal: false,
            frag,
            error: new Error(`Timeout after ${loaderConfig.timeout}ms`),
            networkDetails,
            stats
          }));
        },
        onProgress: (stats, context, data, networkDetails) => {
          if (onProgress) {
            onProgress({
              frag,
              part: null,
              payload: data,
              networkDetails
            });
          }
        }
      });
    });
  }
  loadPart(frag, part, onProgress) {
    this.abort();
    const config = this.config;
    const FragmentILoader = config.fLoader;
    const DefaultILoader = config.loader;
    return new Promise((resolve, reject) => {
      if (this.loader) {
        this.loader.destroy();
      }
      if (frag.gap || part.gap) {
        reject(createGapLoadError(frag, part));
        return;
      }
      const loader = this.loader = frag.loader = FragmentILoader ? new FragmentILoader(config) : new DefaultILoader(config);
      const loaderContext = createLoaderContext(frag, part);
      const loadPolicy = getLoaderConfigWithoutReties(config.fragLoadPolicy.default);
      const loaderConfig = {
        loadPolicy,
        timeout: loadPolicy.maxLoadTimeMs,
        maxRetry: 0,
        retryDelay: 0,
        maxRetryDelay: 0,
        highWaterMark: MIN_CHUNK_SIZE
      };
      part.stats = loader.stats;
      loader.load(loaderContext, loaderConfig, {
        onSuccess: (response, stats, context, networkDetails) => {
          this.resetLoader(frag, loader);
          this.updateStatsFromPart(frag, part);
          const partLoadedData = {
            frag,
            part,
            payload: response.data,
            networkDetails
          };
          onProgress(partLoadedData);
          resolve(partLoadedData);
        },
        onError: (response, context, networkDetails, stats) => {
          this.resetLoader(frag, loader);
          reject(new LoadError({
            type: ErrorTypes.NETWORK_ERROR,
            details: ErrorDetails.FRAG_LOAD_ERROR,
            fatal: false,
            frag,
            part,
            response: _objectSpread2({
              url: loaderContext.url,
              data: void 0
            }, response),
            error: new Error(`HTTP Error ${response.code} ${response.text}`),
            networkDetails,
            stats
          }));
        },
        onAbort: (stats, context, networkDetails) => {
          frag.stats.aborted = part.stats.aborted;
          this.resetLoader(frag, loader);
          reject(new LoadError({
            type: ErrorTypes.NETWORK_ERROR,
            details: ErrorDetails.INTERNAL_ABORTED,
            fatal: false,
            frag,
            part,
            error: new Error("Aborted"),
            networkDetails,
            stats
          }));
        },
        onTimeout: (stats, context, networkDetails) => {
          this.resetLoader(frag, loader);
          reject(new LoadError({
            type: ErrorTypes.NETWORK_ERROR,
            details: ErrorDetails.FRAG_LOAD_TIMEOUT,
            fatal: false,
            frag,
            part,
            error: new Error(`Timeout after ${loaderConfig.timeout}ms`),
            networkDetails,
            stats
          }));
        }
      });
    });
  }
  updateStatsFromPart(frag, part) {
    const fragStats = frag.stats;
    const partStats = part.stats;
    const partTotal = partStats.total;
    fragStats.loaded += partStats.loaded;
    if (partTotal) {
      const estTotalParts = Math.round(frag.duration / part.duration);
      const estLoadedParts = Math.min(Math.round(fragStats.loaded / partTotal), estTotalParts);
      const estRemainingParts = estTotalParts - estLoadedParts;
      const estRemainingBytes = estRemainingParts * Math.round(fragStats.loaded / estLoadedParts);
      fragStats.total = fragStats.loaded + estRemainingBytes;
    } else {
      fragStats.total = Math.max(fragStats.loaded, fragStats.total);
    }
    const fragLoading = fragStats.loading;
    const partLoading = partStats.loading;
    if (fragLoading.start) {
      fragLoading.first += partLoading.first - partLoading.start;
    } else {
      fragLoading.start = partLoading.start;
      fragLoading.first = partLoading.first;
    }
    fragLoading.end = partLoading.end;
  }
  resetLoader(frag, loader) {
    frag.loader = null;
    if (this.loader === loader) {
      self.clearTimeout(this.partLoadTimeout);
      this.loader = null;
    }
    loader.destroy();
  }
}
function createLoaderContext(frag, part = null) {
  const segment = part || frag;
  const loaderContext = {
    frag,
    part,
    responseType: "arraybuffer",
    url: segment.url,
    headers: {},
    rangeStart: 0,
    rangeEnd: 0
  };
  const start = segment.byteRangeStartOffset;
  const end = segment.byteRangeEndOffset;
  if (isFiniteNumber(start) && isFiniteNumber(end)) {
    var _frag$decryptdata;
    let byteRangeStart = start;
    let byteRangeEnd = end;
    if (frag.sn === "initSegment" && ((_frag$decryptdata = frag.decryptdata) == null ? void 0 : _frag$decryptdata.method) === "AES-128") {
      const fragmentLen = end - start;
      if (fragmentLen % 16) {
        byteRangeEnd = end + (16 - fragmentLen % 16);
      }
      if (start !== 0) {
        loaderContext.resetIV = true;
        byteRangeStart = start - 16;
      }
    }
    loaderContext.rangeStart = byteRangeStart;
    loaderContext.rangeEnd = byteRangeEnd;
  }
  return loaderContext;
}
function createGapLoadError(frag, part) {
  const error = new Error(`GAP ${frag.gap ? "tag" : "attribute"} found`);
  const errorData = {
    type: ErrorTypes.MEDIA_ERROR,
    details: ErrorDetails.FRAG_GAP,
    fatal: false,
    frag,
    error,
    networkDetails: null
  };
  if (part) {
    errorData.part = part;
  }
  (part ? part : frag).stats.aborted = true;
  return new LoadError(errorData);
}
class LoadError extends Error {
  constructor(data) {
    super(data.error.message);
    this.data = void 0;
    this.data = data;
  }
}
class AESCrypto {
  constructor(subtle, iv) {
    this.subtle = void 0;
    this.aesIV = void 0;
    this.subtle = subtle;
    this.aesIV = iv;
  }
  decrypt(data, key) {
    return this.subtle.decrypt({
      name: "AES-CBC",
      iv: this.aesIV
    }, key, data);
  }
}
class FastAESKey {
  constructor(subtle, key) {
    this.subtle = void 0;
    this.key = void 0;
    this.subtle = subtle;
    this.key = key;
  }
  expandKey() {
    return this.subtle.importKey("raw", this.key, {
      name: "AES-CBC"
    }, false, ["encrypt", "decrypt"]);
  }
}
function removePadding(array) {
  const outputBytes = array.byteLength;
  const paddingBytes = outputBytes && new DataView(array.buffer).getUint8(outputBytes - 1);
  if (paddingBytes) {
    return sliceUint8(array, 0, outputBytes - paddingBytes);
  }
  return array;
}
class AESDecryptor {
  constructor() {
    this.rcon = [0, 1, 2, 4, 8, 16, 32, 64, 128, 27, 54];
    this.subMix = [new Uint32Array(256), new Uint32Array(256), new Uint32Array(256), new Uint32Array(256)];
    this.invSubMix = [new Uint32Array(256), new Uint32Array(256), new Uint32Array(256), new Uint32Array(256)];
    this.sBox = new Uint32Array(256);
    this.invSBox = new Uint32Array(256);
    this.key = new Uint32Array(0);
    this.ksRows = 0;
    this.keySize = 0;
    this.keySchedule = void 0;
    this.invKeySchedule = void 0;
    this.initTable();
  }
  // Using view.getUint32() also swaps the byte order.
  uint8ArrayToUint32Array_(arrayBuffer) {
    const view = new DataView(arrayBuffer);
    const newArray = new Uint32Array(4);
    for (let i2 = 0; i2 < 4; i2++) {
      newArray[i2] = view.getUint32(i2 * 4);
    }
    return newArray;
  }
  initTable() {
    const sBox = this.sBox;
    const invSBox = this.invSBox;
    const subMix = this.subMix;
    const subMix0 = subMix[0];
    const subMix1 = subMix[1];
    const subMix2 = subMix[2];
    const subMix3 = subMix[3];
    const invSubMix = this.invSubMix;
    const invSubMix0 = invSubMix[0];
    const invSubMix1 = invSubMix[1];
    const invSubMix2 = invSubMix[2];
    const invSubMix3 = invSubMix[3];
    const d = new Uint32Array(256);
    let x = 0;
    let xi = 0;
    let i2 = 0;
    for (i2 = 0; i2 < 256; i2++) {
      if (i2 < 128) {
        d[i2] = i2 << 1;
      } else {
        d[i2] = i2 << 1 ^ 283;
      }
    }
    for (i2 = 0; i2 < 256; i2++) {
      let sx = xi ^ xi << 1 ^ xi << 2 ^ xi << 3 ^ xi << 4;
      sx = sx >>> 8 ^ sx & 255 ^ 99;
      sBox[x] = sx;
      invSBox[sx] = x;
      const x2 = d[x];
      const x4 = d[x2];
      const x8 = d[x4];
      let t2 = d[sx] * 257 ^ sx * 16843008;
      subMix0[x] = t2 << 24 | t2 >>> 8;
      subMix1[x] = t2 << 16 | t2 >>> 16;
      subMix2[x] = t2 << 8 | t2 >>> 24;
      subMix3[x] = t2;
      t2 = x8 * 16843009 ^ x4 * 65537 ^ x2 * 257 ^ x * 16843008;
      invSubMix0[sx] = t2 << 24 | t2 >>> 8;
      invSubMix1[sx] = t2 << 16 | t2 >>> 16;
      invSubMix2[sx] = t2 << 8 | t2 >>> 24;
      invSubMix3[sx] = t2;
      if (!x) {
        x = xi = 1;
      } else {
        x = x2 ^ d[d[d[x8 ^ x2]]];
        xi ^= d[d[xi]];
      }
    }
  }
  expandKey(keyBuffer) {
    const key = this.uint8ArrayToUint32Array_(keyBuffer);
    let sameKey = true;
    let offset = 0;
    while (offset < key.length && sameKey) {
      sameKey = key[offset] === this.key[offset];
      offset++;
    }
    if (sameKey) {
      return;
    }
    this.key = key;
    const keySize = this.keySize = key.length;
    if (keySize !== 4 && keySize !== 6 && keySize !== 8) {
      throw new Error("Invalid aes key size=" + keySize);
    }
    const ksRows = this.ksRows = (keySize + 6 + 1) * 4;
    let ksRow;
    let invKsRow;
    const keySchedule = this.keySchedule = new Uint32Array(ksRows);
    const invKeySchedule = this.invKeySchedule = new Uint32Array(ksRows);
    const sbox = this.sBox;
    const rcon = this.rcon;
    const invSubMix = this.invSubMix;
    const invSubMix0 = invSubMix[0];
    const invSubMix1 = invSubMix[1];
    const invSubMix2 = invSubMix[2];
    const invSubMix3 = invSubMix[3];
    let prev;
    let t2;
    for (ksRow = 0; ksRow < ksRows; ksRow++) {
      if (ksRow < keySize) {
        prev = keySchedule[ksRow] = key[ksRow];
        continue;
      }
      t2 = prev;
      if (ksRow % keySize === 0) {
        t2 = t2 << 8 | t2 >>> 24;
        t2 = sbox[t2 >>> 24] << 24 | sbox[t2 >>> 16 & 255] << 16 | sbox[t2 >>> 8 & 255] << 8 | sbox[t2 & 255];
        t2 ^= rcon[ksRow / keySize | 0] << 24;
      } else if (keySize > 6 && ksRow % keySize === 4) {
        t2 = sbox[t2 >>> 24] << 24 | sbox[t2 >>> 16 & 255] << 16 | sbox[t2 >>> 8 & 255] << 8 | sbox[t2 & 255];
      }
      keySchedule[ksRow] = prev = (keySchedule[ksRow - keySize] ^ t2) >>> 0;
    }
    for (invKsRow = 0; invKsRow < ksRows; invKsRow++) {
      ksRow = ksRows - invKsRow;
      if (invKsRow & 3) {
        t2 = keySchedule[ksRow];
      } else {
        t2 = keySchedule[ksRow - 4];
      }
      if (invKsRow < 4 || ksRow <= 4) {
        invKeySchedule[invKsRow] = t2;
      } else {
        invKeySchedule[invKsRow] = invSubMix0[sbox[t2 >>> 24]] ^ invSubMix1[sbox[t2 >>> 16 & 255]] ^ invSubMix2[sbox[t2 >>> 8 & 255]] ^ invSubMix3[sbox[t2 & 255]];
      }
      invKeySchedule[invKsRow] = invKeySchedule[invKsRow] >>> 0;
    }
  }
  // Adding this as a method greatly improves performance.
  networkToHostOrderSwap(word) {
    return word << 24 | (word & 65280) << 8 | (word & 16711680) >> 8 | word >>> 24;
  }
  decrypt(inputArrayBuffer, offset, aesIV) {
    const nRounds = this.keySize + 6;
    const invKeySchedule = this.invKeySchedule;
    const invSBOX = this.invSBox;
    const invSubMix = this.invSubMix;
    const invSubMix0 = invSubMix[0];
    const invSubMix1 = invSubMix[1];
    const invSubMix2 = invSubMix[2];
    const invSubMix3 = invSubMix[3];
    const initVector = this.uint8ArrayToUint32Array_(aesIV);
    let initVector0 = initVector[0];
    let initVector1 = initVector[1];
    let initVector2 = initVector[2];
    let initVector3 = initVector[3];
    const inputInt32 = new Int32Array(inputArrayBuffer);
    const outputInt32 = new Int32Array(inputInt32.length);
    let t0, t1, t2, t3;
    let s0, s1, s2, s3;
    let inputWords0, inputWords1, inputWords2, inputWords3;
    let ksRow, i2;
    const swapWord = this.networkToHostOrderSwap;
    while (offset < inputInt32.length) {
      inputWords0 = swapWord(inputInt32[offset]);
      inputWords1 = swapWord(inputInt32[offset + 1]);
      inputWords2 = swapWord(inputInt32[offset + 2]);
      inputWords3 = swapWord(inputInt32[offset + 3]);
      s0 = inputWords0 ^ invKeySchedule[0];
      s1 = inputWords3 ^ invKeySchedule[1];
      s2 = inputWords2 ^ invKeySchedule[2];
      s3 = inputWords1 ^ invKeySchedule[3];
      ksRow = 4;
      for (i2 = 1; i2 < nRounds; i2++) {
        t0 = invSubMix0[s0 >>> 24] ^ invSubMix1[s1 >> 16 & 255] ^ invSubMix2[s2 >> 8 & 255] ^ invSubMix3[s3 & 255] ^ invKeySchedule[ksRow];
        t1 = invSubMix0[s1 >>> 24] ^ invSubMix1[s2 >> 16 & 255] ^ invSubMix2[s3 >> 8 & 255] ^ invSubMix3[s0 & 255] ^ invKeySchedule[ksRow + 1];
        t2 = invSubMix0[s2 >>> 24] ^ invSubMix1[s3 >> 16 & 255] ^ invSubMix2[s0 >> 8 & 255] ^ invSubMix3[s1 & 255] ^ invKeySchedule[ksRow + 2];
        t3 = invSubMix0[s3 >>> 24] ^ invSubMix1[s0 >> 16 & 255] ^ invSubMix2[s1 >> 8 & 255] ^ invSubMix3[s2 & 255] ^ invKeySchedule[ksRow + 3];
        s0 = t0;
        s1 = t1;
        s2 = t2;
        s3 = t3;
        ksRow = ksRow + 4;
      }
      t0 = invSBOX[s0 >>> 24] << 24 ^ invSBOX[s1 >> 16 & 255] << 16 ^ invSBOX[s2 >> 8 & 255] << 8 ^ invSBOX[s3 & 255] ^ invKeySchedule[ksRow];
      t1 = invSBOX[s1 >>> 24] << 24 ^ invSBOX[s2 >> 16 & 255] << 16 ^ invSBOX[s3 >> 8 & 255] << 8 ^ invSBOX[s0 & 255] ^ invKeySchedule[ksRow + 1];
      t2 = invSBOX[s2 >>> 24] << 24 ^ invSBOX[s3 >> 16 & 255] << 16 ^ invSBOX[s0 >> 8 & 255] << 8 ^ invSBOX[s1 & 255] ^ invKeySchedule[ksRow + 2];
      t3 = invSBOX[s3 >>> 24] << 24 ^ invSBOX[s0 >> 16 & 255] << 16 ^ invSBOX[s1 >> 8 & 255] << 8 ^ invSBOX[s2 & 255] ^ invKeySchedule[ksRow + 3];
      outputInt32[offset] = swapWord(t0 ^ initVector0);
      outputInt32[offset + 1] = swapWord(t3 ^ initVector1);
      outputInt32[offset + 2] = swapWord(t2 ^ initVector2);
      outputInt32[offset + 3] = swapWord(t1 ^ initVector3);
      initVector0 = inputWords0;
      initVector1 = inputWords1;
      initVector2 = inputWords2;
      initVector3 = inputWords3;
      offset = offset + 4;
    }
    return outputInt32.buffer;
  }
}
const CHUNK_SIZE = 16;
class Decrypter {
  constructor(config, {
    removePKCS7Padding = true
  } = {}) {
    this.logEnabled = true;
    this.removePKCS7Padding = void 0;
    this.subtle = null;
    this.softwareDecrypter = null;
    this.key = null;
    this.fastAesKey = null;
    this.remainderData = null;
    this.currentIV = null;
    this.currentResult = null;
    this.useSoftware = void 0;
    this.useSoftware = config.enableSoftwareAES;
    this.removePKCS7Padding = removePKCS7Padding;
    if (removePKCS7Padding) {
      try {
        const browserCrypto = self.crypto;
        if (browserCrypto) {
          this.subtle = browserCrypto.subtle || browserCrypto.webkitSubtle;
        }
      } catch (e2) {
      }
    }
    this.useSoftware = !this.subtle;
  }
  destroy() {
    this.subtle = null;
    this.softwareDecrypter = null;
    this.key = null;
    this.fastAesKey = null;
    this.remainderData = null;
    this.currentIV = null;
    this.currentResult = null;
  }
  isSync() {
    return this.useSoftware;
  }
  flush() {
    const {
      currentResult,
      remainderData
    } = this;
    if (!currentResult || remainderData) {
      this.reset();
      return null;
    }
    const data = new Uint8Array(currentResult);
    this.reset();
    if (this.removePKCS7Padding) {
      return removePadding(data);
    }
    return data;
  }
  reset() {
    this.currentResult = null;
    this.currentIV = null;
    this.remainderData = null;
    if (this.softwareDecrypter) {
      this.softwareDecrypter = null;
    }
  }
  decrypt(data, key, iv) {
    if (this.useSoftware) {
      return new Promise((resolve, reject) => {
        this.softwareDecrypt(new Uint8Array(data), key, iv);
        const decryptResult = this.flush();
        if (decryptResult) {
          resolve(decryptResult.buffer);
        } else {
          reject(new Error("[softwareDecrypt] Failed to decrypt data"));
        }
      });
    }
    return this.webCryptoDecrypt(new Uint8Array(data), key, iv);
  }
  // Software decryption is progressive. Progressive decryption may not return a result on each call. Any cached
  // data is handled in the flush() call
  softwareDecrypt(data, key, iv) {
    const {
      currentIV,
      currentResult,
      remainderData
    } = this;
    this.logOnce("JS AES decrypt");
    if (remainderData) {
      data = appendUint8Array(remainderData, data);
      this.remainderData = null;
    }
    const currentChunk = this.getValidChunk(data);
    if (!currentChunk.length) {
      return null;
    }
    if (currentIV) {
      iv = currentIV;
    }
    let softwareDecrypter = this.softwareDecrypter;
    if (!softwareDecrypter) {
      softwareDecrypter = this.softwareDecrypter = new AESDecryptor();
    }
    softwareDecrypter.expandKey(key);
    const result = currentResult;
    this.currentResult = softwareDecrypter.decrypt(currentChunk.buffer, 0, iv);
    this.currentIV = sliceUint8(currentChunk, -16).buffer;
    if (!result) {
      return null;
    }
    return result;
  }
  webCryptoDecrypt(data, key, iv) {
    if (this.key !== key || !this.fastAesKey) {
      if (!this.subtle) {
        return Promise.resolve(this.onWebCryptoError(data, key, iv));
      }
      this.key = key;
      this.fastAesKey = new FastAESKey(this.subtle, key);
    }
    return this.fastAesKey.expandKey().then((aesKey) => {
      if (!this.subtle) {
        return Promise.reject(new Error("web crypto not initialized"));
      }
      this.logOnce("WebCrypto AES decrypt");
      const crypto2 = new AESCrypto(this.subtle, new Uint8Array(iv));
      return crypto2.decrypt(data.buffer, aesKey);
    }).catch((err) => {
      logger.warn(`[decrypter]: WebCrypto Error, disable WebCrypto API, ${err.name}: ${err.message}`);
      return this.onWebCryptoError(data, key, iv);
    });
  }
  onWebCryptoError(data, key, iv) {
    this.useSoftware = true;
    this.logEnabled = true;
    this.softwareDecrypt(data, key, iv);
    const decryptResult = this.flush();
    if (decryptResult) {
      return decryptResult.buffer;
    }
    throw new Error("WebCrypto and softwareDecrypt: failed to decrypt data");
  }
  getValidChunk(data) {
    let currentChunk = data;
    const splitPoint = data.length - data.length % CHUNK_SIZE;
    if (splitPoint !== data.length) {
      currentChunk = sliceUint8(data, 0, splitPoint);
      this.remainderData = sliceUint8(data, splitPoint);
    }
    return currentChunk;
  }
  logOnce(msg) {
    if (!this.logEnabled) {
      return;
    }
    logger.log(`[decrypter]: ${msg}`);
    this.logEnabled = false;
  }
}
const TimeRanges = {
  toString: function(r2) {
    let log = "";
    const len = r2.length;
    for (let i2 = 0; i2 < len; i2++) {
      log += `[${r2.start(i2).toFixed(3)}-${r2.end(i2).toFixed(3)}]`;
    }
    return log;
  }
};
const State = {
  STOPPED: "STOPPED",
  IDLE: "IDLE",
  KEY_LOADING: "KEY_LOADING",
  FRAG_LOADING: "FRAG_LOADING",
  FRAG_LOADING_WAITING_RETRY: "FRAG_LOADING_WAITING_RETRY",
  WAITING_TRACK: "WAITING_TRACK",
  PARSING: "PARSING",
  PARSED: "PARSED",
  ENDED: "ENDED",
  ERROR: "ERROR",
  WAITING_INIT_PTS: "WAITING_INIT_PTS",
  WAITING_LEVEL: "WAITING_LEVEL"
};
class BaseStreamController extends TaskLoop {
  constructor(hls, fragmentTracker, keyLoader, logPrefix, playlistType) {
    super();
    this.hls = void 0;
    this.fragPrevious = null;
    this.fragCurrent = null;
    this.fragmentTracker = void 0;
    this.transmuxer = null;
    this._state = State.STOPPED;
    this.playlistType = void 0;
    this.media = null;
    this.mediaBuffer = null;
    this.config = void 0;
    this.bitrateTest = false;
    this.lastCurrentTime = 0;
    this.nextLoadPosition = 0;
    this.startPosition = 0;
    this.startTimeOffset = null;
    this.loadedmetadata = false;
    this.retryDate = 0;
    this.levels = null;
    this.fragmentLoader = void 0;
    this.keyLoader = void 0;
    this.levelLastLoaded = null;
    this.startFragRequested = false;
    this.decrypter = void 0;
    this.initPTS = [];
    this.onvseeking = null;
    this.onvended = null;
    this.logPrefix = "";
    this.log = void 0;
    this.warn = void 0;
    this.playlistType = playlistType;
    this.logPrefix = logPrefix;
    this.log = logger.log.bind(logger, `${logPrefix}:`);
    this.warn = logger.warn.bind(logger, `${logPrefix}:`);
    this.hls = hls;
    this.fragmentLoader = new FragmentLoader(hls.config);
    this.keyLoader = keyLoader;
    this.fragmentTracker = fragmentTracker;
    this.config = hls.config;
    this.decrypter = new Decrypter(hls.config);
    hls.on(Events.MANIFEST_LOADED, this.onManifestLoaded, this);
  }
  doTick() {
    this.onTickEnd();
  }
  onTickEnd() {
  }
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  startLoad(startPosition) {
  }
  stopLoad() {
    this.fragmentLoader.abort();
    this.keyLoader.abort(this.playlistType);
    const frag = this.fragCurrent;
    if (frag != null && frag.loader) {
      frag.abortRequests();
      this.fragmentTracker.removeFragment(frag);
    }
    this.resetTransmuxer();
    this.fragCurrent = null;
    this.fragPrevious = null;
    this.clearInterval();
    this.clearNextTick();
    this.state = State.STOPPED;
  }
  _streamEnded(bufferInfo, levelDetails) {
    if (levelDetails.live || bufferInfo.nextStart || !bufferInfo.end || !this.media) {
      return false;
    }
    const partList = levelDetails.partList;
    if (partList != null && partList.length) {
      const lastPart = partList[partList.length - 1];
      const lastPartBuffered = BufferHelper.isBuffered(this.media, lastPart.start + lastPart.duration / 2);
      return lastPartBuffered;
    }
    const playlistType = levelDetails.fragments[levelDetails.fragments.length - 1].type;
    return this.fragmentTracker.isEndListAppended(playlistType);
  }
  getLevelDetails() {
    if (this.levels && this.levelLastLoaded !== null) {
      var _this$levelLastLoaded;
      return (_this$levelLastLoaded = this.levelLastLoaded) == null ? void 0 : _this$levelLastLoaded.details;
    }
  }
  onMediaAttached(event, data) {
    const media = this.media = this.mediaBuffer = data.media;
    this.onvseeking = this.onMediaSeeking.bind(this);
    this.onvended = this.onMediaEnded.bind(this);
    media.addEventListener("seeking", this.onvseeking);
    media.addEventListener("ended", this.onvended);
    const config = this.config;
    if (this.levels && config.autoStartLoad && this.state === State.STOPPED) {
      this.startLoad(config.startPosition);
    }
  }
  onMediaDetaching() {
    const media = this.media;
    if (media != null && media.ended) {
      this.log("MSE detaching and video ended, reset startPosition");
      this.startPosition = this.lastCurrentTime = 0;
    }
    if (media && this.onvseeking && this.onvended) {
      media.removeEventListener("seeking", this.onvseeking);
      media.removeEventListener("ended", this.onvended);
      this.onvseeking = this.onvended = null;
    }
    if (this.keyLoader) {
      this.keyLoader.detach();
    }
    this.media = this.mediaBuffer = null;
    this.loadedmetadata = false;
    this.fragmentTracker.removeAllFragments();
    this.stopLoad();
  }
  onMediaSeeking() {
    const {
      config,
      fragCurrent,
      media,
      mediaBuffer,
      state
    } = this;
    const currentTime = media ? media.currentTime : 0;
    const bufferInfo = BufferHelper.bufferInfo(mediaBuffer ? mediaBuffer : media, currentTime, config.maxBufferHole);
    this.log(`media seeking to ${isFiniteNumber(currentTime) ? currentTime.toFixed(3) : currentTime}, state: ${state}`);
    if (this.state === State.ENDED) {
      this.resetLoadingState();
    } else if (fragCurrent) {
      const tolerance = config.maxFragLookUpTolerance;
      const fragStartOffset = fragCurrent.start - tolerance;
      const fragEndOffset = fragCurrent.start + fragCurrent.duration + tolerance;
      if (!bufferInfo.len || fragEndOffset < bufferInfo.start || fragStartOffset > bufferInfo.end) {
        const pastFragment = currentTime > fragEndOffset;
        if (currentTime < fragStartOffset || pastFragment) {
          if (pastFragment && fragCurrent.loader) {
            this.log("seeking outside of buffer while fragment load in progress, cancel fragment load");
            fragCurrent.abortRequests();
            this.resetLoadingState();
          }
          this.fragPrevious = null;
        }
      }
    }
    if (media) {
      this.fragmentTracker.removeFragmentsInRange(currentTime, Infinity, this.playlistType, true);
      this.lastCurrentTime = currentTime;
    }
    if (!this.loadedmetadata && !bufferInfo.len) {
      this.nextLoadPosition = this.startPosition = currentTime;
    }
    this.tickImmediate();
  }
  onMediaEnded() {
    this.startPosition = this.lastCurrentTime = 0;
  }
  onManifestLoaded(event, data) {
    this.startTimeOffset = data.startTimeOffset;
    this.initPTS = [];
  }
  onHandlerDestroying() {
    this.hls.off(Events.MANIFEST_LOADED, this.onManifestLoaded, this);
    this.stopLoad();
    super.onHandlerDestroying();
    this.hls = null;
  }
  onHandlerDestroyed() {
    this.state = State.STOPPED;
    if (this.fragmentLoader) {
      this.fragmentLoader.destroy();
    }
    if (this.keyLoader) {
      this.keyLoader.destroy();
    }
    if (this.decrypter) {
      this.decrypter.destroy();
    }
    this.hls = this.log = this.warn = this.decrypter = this.keyLoader = this.fragmentLoader = this.fragmentTracker = null;
    super.onHandlerDestroyed();
  }
  loadFragment(frag, level, targetBufferTime) {
    this._loadFragForPlayback(frag, level, targetBufferTime);
  }
  _loadFragForPlayback(frag, level, targetBufferTime) {
    const progressCallback = (data) => {
      if (this.fragContextChanged(frag)) {
        this.warn(`Fragment ${frag.sn}${data.part ? " p: " + data.part.index : ""} of level ${frag.level} was dropped during download.`);
        this.fragmentTracker.removeFragment(frag);
        return;
      }
      frag.stats.chunkCount++;
      this._handleFragmentLoadProgress(data);
    };
    this._doFragLoad(frag, level, targetBufferTime, progressCallback).then((data) => {
      if (!data) {
        return;
      }
      const state = this.state;
      if (this.fragContextChanged(frag)) {
        if (state === State.FRAG_LOADING || !this.fragCurrent && state === State.PARSING) {
          this.fragmentTracker.removeFragment(frag);
          this.state = State.IDLE;
        }
        return;
      }
      if ("payload" in data) {
        this.log(`Loaded fragment ${frag.sn} of level ${frag.level}`);
        this.hls.trigger(Events.FRAG_LOADED, data);
      }
      this._handleFragmentLoadComplete(data);
    }).catch((reason) => {
      if (this.state === State.STOPPED || this.state === State.ERROR) {
        return;
      }
      this.warn(`Frag error: ${(reason == null ? void 0 : reason.message) || reason}`);
      this.resetFragmentLoading(frag);
    });
  }
  clearTrackerIfNeeded(frag) {
    var _this$mediaBuffer;
    const {
      fragmentTracker
    } = this;
    const fragState = fragmentTracker.getState(frag);
    if (fragState === FragmentState.APPENDING) {
      const playlistType = frag.type;
      const bufferedInfo = this.getFwdBufferInfo(this.mediaBuffer, playlistType);
      const minForwardBufferLength = Math.max(frag.duration, bufferedInfo ? bufferedInfo.len : this.config.maxBufferLength);
      const backtrackFragment = this.backtrackFragment;
      const backtracked = backtrackFragment ? frag.sn - backtrackFragment.sn : 0;
      if (backtracked === 1 || this.reduceMaxBufferLength(minForwardBufferLength, frag.duration)) {
        fragmentTracker.removeFragment(frag);
      }
    } else if (((_this$mediaBuffer = this.mediaBuffer) == null ? void 0 : _this$mediaBuffer.buffered.length) === 0) {
      fragmentTracker.removeAllFragments();
    } else if (fragmentTracker.hasParts(frag.type)) {
      fragmentTracker.detectPartialFragments({
        frag,
        part: null,
        stats: frag.stats,
        id: frag.type
      });
      if (fragmentTracker.getState(frag) === FragmentState.PARTIAL) {
        fragmentTracker.removeFragment(frag);
      }
    }
  }
  checkLiveUpdate(details) {
    if (details.updated && !details.live) {
      const lastFragment = details.fragments[details.fragments.length - 1];
      this.fragmentTracker.detectPartialFragments({
        frag: lastFragment,
        part: null,
        stats: lastFragment.stats,
        id: lastFragment.type
      });
    }
    if (!details.fragments[0]) {
      details.deltaUpdateFailed = true;
    }
  }
  flushMainBuffer(startOffset, endOffset, type = null) {
    if (!(startOffset - endOffset)) {
      return;
    }
    const flushScope = {
      startOffset,
      endOffset,
      type
    };
    this.hls.trigger(Events.BUFFER_FLUSHING, flushScope);
  }
  _loadInitSegment(frag, level) {
    this._doFragLoad(frag, level).then((data) => {
      if (!data || this.fragContextChanged(frag) || !this.levels) {
        throw new Error("init load aborted");
      }
      return data;
    }).then((data) => {
      const {
        hls
      } = this;
      const {
        payload
      } = data;
      const decryptData = frag.decryptdata;
      if (payload && payload.byteLength > 0 && decryptData != null && decryptData.key && decryptData.iv && decryptData.method === "AES-128") {
        const startTime = self.performance.now();
        return this.decrypter.decrypt(new Uint8Array(payload), decryptData.key.buffer, decryptData.iv.buffer).catch((err) => {
          hls.trigger(Events.ERROR, {
            type: ErrorTypes.MEDIA_ERROR,
            details: ErrorDetails.FRAG_DECRYPT_ERROR,
            fatal: false,
            error: err,
            reason: err.message,
            frag
          });
          throw err;
        }).then((decryptedData) => {
          const endTime = self.performance.now();
          hls.trigger(Events.FRAG_DECRYPTED, {
            frag,
            payload: decryptedData,
            stats: {
              tstart: startTime,
              tdecrypt: endTime
            }
          });
          data.payload = decryptedData;
          return this.completeInitSegmentLoad(data);
        });
      }
      return this.completeInitSegmentLoad(data);
    }).catch((reason) => {
      if (this.state === State.STOPPED || this.state === State.ERROR) {
        return;
      }
      this.warn(reason);
      this.resetFragmentLoading(frag);
    });
  }
  completeInitSegmentLoad(data) {
    const {
      levels
    } = this;
    if (!levels) {
      throw new Error("init load aborted, missing levels");
    }
    const stats = data.frag.stats;
    this.state = State.IDLE;
    data.frag.data = new Uint8Array(data.payload);
    stats.parsing.start = stats.buffering.start = self.performance.now();
    stats.parsing.end = stats.buffering.end = self.performance.now();
    this.tick();
  }
  fragContextChanged(frag) {
    const {
      fragCurrent
    } = this;
    return !frag || !fragCurrent || frag.sn !== fragCurrent.sn || frag.level !== fragCurrent.level;
  }
  fragBufferedComplete(frag, part) {
    var _frag$startPTS, _frag$endPTS, _this$fragCurrent, _this$fragPrevious;
    const media = this.mediaBuffer ? this.mediaBuffer : this.media;
    this.log(`Buffered ${frag.type} sn: ${frag.sn}${part ? " part: " + part.index : ""} of ${this.playlistType === PlaylistLevelType.MAIN ? "level" : "track"} ${frag.level} (frag:[${((_frag$startPTS = frag.startPTS) != null ? _frag$startPTS : NaN).toFixed(3)}-${((_frag$endPTS = frag.endPTS) != null ? _frag$endPTS : NaN).toFixed(3)}] > buffer:${media ? TimeRanges.toString(BufferHelper.getBuffered(media)) : "(detached)"})`);
    if (frag.sn !== "initSegment") {
      var _this$levels;
      if (frag.type !== PlaylistLevelType.SUBTITLE) {
        const el = frag.elementaryStreams;
        if (!Object.keys(el).some((type) => !!el[type])) {
          this.state = State.IDLE;
          return;
        }
      }
      const level = (_this$levels = this.levels) == null ? void 0 : _this$levels[frag.level];
      if (level != null && level.fragmentError) {
        this.log(`Resetting level fragment error count of ${level.fragmentError} on frag buffered`);
        level.fragmentError = 0;
      }
    }
    this.state = State.IDLE;
    if (!media) {
      return;
    }
    if (!this.loadedmetadata && frag.type == PlaylistLevelType.MAIN && media.buffered.length && ((_this$fragCurrent = this.fragCurrent) == null ? void 0 : _this$fragCurrent.sn) === ((_this$fragPrevious = this.fragPrevious) == null ? void 0 : _this$fragPrevious.sn)) {
      this.loadedmetadata = true;
      this.seekToStartPos();
    }
    this.tick();
  }
  seekToStartPos() {
  }
  _handleFragmentLoadComplete(fragLoadedEndData) {
    const {
      transmuxer
    } = this;
    if (!transmuxer) {
      return;
    }
    const {
      frag,
      part,
      partsLoaded
    } = fragLoadedEndData;
    const complete = !partsLoaded || partsLoaded.length === 0 || partsLoaded.some((fragLoaded) => !fragLoaded);
    const chunkMeta = new ChunkMetadata(frag.level, frag.sn, frag.stats.chunkCount + 1, 0, part ? part.index : -1, !complete);
    transmuxer.flush(chunkMeta);
  }
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  _handleFragmentLoadProgress(frag) {
  }
  _doFragLoad(frag, level, targetBufferTime = null, progressCallback) {
    var _frag$decryptdata;
    const details = level == null ? void 0 : level.details;
    if (!this.levels || !details) {
      throw new Error(`frag load aborted, missing level${details ? "" : " detail"}s`);
    }
    let keyLoadingPromise = null;
    if (frag.encrypted && !((_frag$decryptdata = frag.decryptdata) != null && _frag$decryptdata.key)) {
      this.log(`Loading key for ${frag.sn} of [${details.startSN}-${details.endSN}], ${this.logPrefix === "[stream-controller]" ? "level" : "track"} ${frag.level}`);
      this.state = State.KEY_LOADING;
      this.fragCurrent = frag;
      keyLoadingPromise = this.keyLoader.load(frag).then((keyLoadedData) => {
        if (!this.fragContextChanged(keyLoadedData.frag)) {
          this.hls.trigger(Events.KEY_LOADED, keyLoadedData);
          if (this.state === State.KEY_LOADING) {
            this.state = State.IDLE;
          }
          return keyLoadedData;
        }
      });
      this.hls.trigger(Events.KEY_LOADING, {
        frag
      });
      if (this.fragCurrent === null) {
        keyLoadingPromise = Promise.reject(new Error(`frag load aborted, context changed in KEY_LOADING`));
      }
    } else if (!frag.encrypted && details.encryptedFragments.length) {
      this.keyLoader.loadClear(frag, details.encryptedFragments);
    }
    targetBufferTime = Math.max(frag.start, targetBufferTime || 0);
    if (this.config.lowLatencyMode && frag.sn !== "initSegment") {
      const partList = details.partList;
      if (partList && progressCallback) {
        if (targetBufferTime > frag.end && details.fragmentHint) {
          frag = details.fragmentHint;
        }
        const partIndex = this.getNextPart(partList, frag, targetBufferTime);
        if (partIndex > -1) {
          const part = partList[partIndex];
          this.log(`Loading part sn: ${frag.sn} p: ${part.index} cc: ${frag.cc} of playlist [${details.startSN}-${details.endSN}] parts [0-${partIndex}-${partList.length - 1}] ${this.logPrefix === "[stream-controller]" ? "level" : "track"}: ${frag.level}, target: ${parseFloat(targetBufferTime.toFixed(3))}`);
          this.nextLoadPosition = part.start + part.duration;
          this.state = State.FRAG_LOADING;
          let _result;
          if (keyLoadingPromise) {
            _result = keyLoadingPromise.then((keyLoadedData) => {
              if (!keyLoadedData || this.fragContextChanged(keyLoadedData.frag)) {
                return null;
              }
              return this.doFragPartsLoad(frag, part, level, progressCallback);
            }).catch((error) => this.handleFragLoadError(error));
          } else {
            _result = this.doFragPartsLoad(frag, part, level, progressCallback).catch((error) => this.handleFragLoadError(error));
          }
          this.hls.trigger(Events.FRAG_LOADING, {
            frag,
            part,
            targetBufferTime
          });
          if (this.fragCurrent === null) {
            return Promise.reject(new Error(`frag load aborted, context changed in FRAG_LOADING parts`));
          }
          return _result;
        } else if (!frag.url || this.loadedEndOfParts(partList, targetBufferTime)) {
          return Promise.resolve(null);
        }
      }
    }
    this.log(`Loading fragment ${frag.sn} cc: ${frag.cc} ${details ? "of [" + details.startSN + "-" + details.endSN + "] " : ""}${this.logPrefix === "[stream-controller]" ? "level" : "track"}: ${frag.level}, target: ${parseFloat(targetBufferTime.toFixed(3))}`);
    if (isFiniteNumber(frag.sn) && !this.bitrateTest) {
      this.nextLoadPosition = frag.start + frag.duration;
    }
    this.state = State.FRAG_LOADING;
    const dataOnProgress = this.config.progressive;
    let result;
    if (dataOnProgress && keyLoadingPromise) {
      result = keyLoadingPromise.then((keyLoadedData) => {
        if (!keyLoadedData || this.fragContextChanged(keyLoadedData == null ? void 0 : keyLoadedData.frag)) {
          return null;
        }
        return this.fragmentLoader.load(frag, progressCallback);
      }).catch((error) => this.handleFragLoadError(error));
    } else {
      result = Promise.all([this.fragmentLoader.load(frag, dataOnProgress ? progressCallback : void 0), keyLoadingPromise]).then(([fragLoadedData]) => {
        if (!dataOnProgress && fragLoadedData && progressCallback) {
          progressCallback(fragLoadedData);
        }
        return fragLoadedData;
      }).catch((error) => this.handleFragLoadError(error));
    }
    this.hls.trigger(Events.FRAG_LOADING, {
      frag,
      targetBufferTime
    });
    if (this.fragCurrent === null) {
      return Promise.reject(new Error(`frag load aborted, context changed in FRAG_LOADING`));
    }
    return result;
  }
  doFragPartsLoad(frag, fromPart, level, progressCallback) {
    return new Promise((resolve, reject) => {
      var _level$details;
      const partsLoaded = [];
      const initialPartList = (_level$details = level.details) == null ? void 0 : _level$details.partList;
      const loadPart = (part) => {
        this.fragmentLoader.loadPart(frag, part, progressCallback).then((partLoadedData) => {
          partsLoaded[part.index] = partLoadedData;
          const loadedPart = partLoadedData.part;
          this.hls.trigger(Events.FRAG_LOADED, partLoadedData);
          const nextPart = getPartWith(level, frag.sn, part.index + 1) || findPart(initialPartList, frag.sn, part.index + 1);
          if (nextPart) {
            loadPart(nextPart);
          } else {
            return resolve({
              frag,
              part: loadedPart,
              partsLoaded
            });
          }
        }).catch(reject);
      };
      loadPart(fromPart);
    });
  }
  handleFragLoadError(error) {
    if ("data" in error) {
      const data = error.data;
      if (error.data && data.details === ErrorDetails.INTERNAL_ABORTED) {
        this.handleFragLoadAborted(data.frag, data.part);
      } else {
        this.hls.trigger(Events.ERROR, data);
      }
    } else {
      this.hls.trigger(Events.ERROR, {
        type: ErrorTypes.OTHER_ERROR,
        details: ErrorDetails.INTERNAL_EXCEPTION,
        err: error,
        error,
        fatal: true
      });
    }
    return null;
  }
  _handleTransmuxerFlush(chunkMeta) {
    const context = this.getCurrentContext(chunkMeta);
    if (!context || this.state !== State.PARSING) {
      if (!this.fragCurrent && this.state !== State.STOPPED && this.state !== State.ERROR) {
        this.state = State.IDLE;
      }
      return;
    }
    const {
      frag,
      part,
      level
    } = context;
    const now2 = self.performance.now();
    frag.stats.parsing.end = now2;
    if (part) {
      part.stats.parsing.end = now2;
    }
    this.updateLevelTiming(frag, part, level, chunkMeta.partial);
  }
  getCurrentContext(chunkMeta) {
    const {
      levels,
      fragCurrent
    } = this;
    const {
      level: levelIndex,
      sn,
      part: partIndex
    } = chunkMeta;
    if (!(levels != null && levels[levelIndex])) {
      this.warn(`Levels object was unset while buffering fragment ${sn} of level ${levelIndex}. The current chunk will not be buffered.`);
      return null;
    }
    const level = levels[levelIndex];
    const part = partIndex > -1 ? getPartWith(level, sn, partIndex) : null;
    const frag = part ? part.fragment : getFragmentWithSN(level, sn, fragCurrent);
    if (!frag) {
      return null;
    }
    if (fragCurrent && fragCurrent !== frag) {
      frag.stats = fragCurrent.stats;
    }
    return {
      frag,
      part,
      level
    };
  }
  bufferFragmentData(data, frag, part, chunkMeta, noBacktracking) {
    var _buffer;
    if (!data || this.state !== State.PARSING) {
      return;
    }
    const {
      data1,
      data2
    } = data;
    let buffer = data1;
    if (data1 && data2) {
      buffer = appendUint8Array(data1, data2);
    }
    if (!((_buffer = buffer) != null && _buffer.length)) {
      return;
    }
    const segment = {
      type: data.type,
      frag,
      part,
      chunkMeta,
      parent: frag.type,
      data: buffer
    };
    this.hls.trigger(Events.BUFFER_APPENDING, segment);
    if (data.dropped && data.independent && !part) {
      if (noBacktracking) {
        return;
      }
      this.flushBufferGap(frag);
    }
  }
  flushBufferGap(frag) {
    const media = this.media;
    if (!media) {
      return;
    }
    if (!BufferHelper.isBuffered(media, media.currentTime)) {
      this.flushMainBuffer(0, frag.start);
      return;
    }
    const currentTime = media.currentTime;
    const bufferInfo = BufferHelper.bufferInfo(media, currentTime, 0);
    const fragDuration = frag.duration;
    const segmentFraction = Math.min(this.config.maxFragLookUpTolerance * 2, fragDuration * 0.25);
    const start = Math.max(Math.min(frag.start - segmentFraction, bufferInfo.end - segmentFraction), currentTime + segmentFraction);
    if (frag.start - start > segmentFraction) {
      this.flushMainBuffer(start, frag.start);
    }
  }
  getFwdBufferInfo(bufferable, type) {
    const pos = this.getLoadPosition();
    if (!isFiniteNumber(pos)) {
      return null;
    }
    return this.getFwdBufferInfoAtPos(bufferable, pos, type);
  }
  getFwdBufferInfoAtPos(bufferable, pos, type) {
    const {
      config: {
        maxBufferHole
      }
    } = this;
    const bufferInfo = BufferHelper.bufferInfo(bufferable, pos, maxBufferHole);
    if (bufferInfo.len === 0 && bufferInfo.nextStart !== void 0) {
      const bufferedFragAtPos = this.fragmentTracker.getBufferedFrag(pos, type);
      if (bufferedFragAtPos && bufferInfo.nextStart < bufferedFragAtPos.end) {
        return BufferHelper.bufferInfo(bufferable, pos, Math.max(bufferInfo.nextStart, maxBufferHole));
      }
    }
    return bufferInfo;
  }
  getMaxBufferLength(levelBitrate) {
    const {
      config
    } = this;
    let maxBufLen;
    if (levelBitrate) {
      maxBufLen = Math.max(8 * config.maxBufferSize / levelBitrate, config.maxBufferLength);
    } else {
      maxBufLen = config.maxBufferLength;
    }
    return Math.min(maxBufLen, config.maxMaxBufferLength);
  }
  reduceMaxBufferLength(threshold, fragDuration) {
    const config = this.config;
    const minLength = Math.max(Math.min(threshold - fragDuration, config.maxBufferLength), fragDuration);
    const reducedLength = Math.max(threshold - fragDuration * 3, config.maxMaxBufferLength / 2, minLength);
    if (reducedLength >= minLength) {
      config.maxMaxBufferLength = reducedLength;
      this.warn(`Reduce max buffer length to ${reducedLength}s`);
      return true;
    }
    return false;
  }
  getAppendedFrag(position, playlistType = PlaylistLevelType.MAIN) {
    const fragOrPart = this.fragmentTracker.getAppendedFrag(position, PlaylistLevelType.MAIN);
    if (fragOrPart && "fragment" in fragOrPart) {
      return fragOrPart.fragment;
    }
    return fragOrPart;
  }
  getNextFragment(pos, levelDetails) {
    const fragments = levelDetails.fragments;
    const fragLen = fragments.length;
    if (!fragLen) {
      return null;
    }
    const {
      config
    } = this;
    const start = fragments[0].start;
    let frag;
    if (levelDetails.live) {
      const initialLiveManifestSize = config.initialLiveManifestSize;
      if (fragLen < initialLiveManifestSize) {
        this.warn(`Not enough fragments to start playback (have: ${fragLen}, need: ${initialLiveManifestSize})`);
        return null;
      }
      if (!levelDetails.PTSKnown && !this.startFragRequested && this.startPosition === -1 || pos < start) {
        frag = this.getInitialLiveFragment(levelDetails, fragments);
        this.startPosition = this.nextLoadPosition = frag ? this.hls.liveSyncPosition || frag.start : pos;
      }
    } else if (pos <= start) {
      frag = fragments[0];
    }
    if (!frag) {
      const end = config.lowLatencyMode ? levelDetails.partEnd : levelDetails.fragmentEnd;
      frag = this.getFragmentAtPosition(pos, end, levelDetails);
    }
    return this.mapToInitFragWhenRequired(frag);
  }
  isLoopLoading(frag, targetBufferTime) {
    const trackerState = this.fragmentTracker.getState(frag);
    return (trackerState === FragmentState.OK || trackerState === FragmentState.PARTIAL && !!frag.gap) && this.nextLoadPosition > targetBufferTime;
  }
  getNextFragmentLoopLoading(frag, levelDetails, bufferInfo, playlistType, maxBufLen) {
    const gapStart = frag.gap;
    const nextFragment = this.getNextFragment(this.nextLoadPosition, levelDetails);
    if (nextFragment === null) {
      return nextFragment;
    }
    frag = nextFragment;
    if (gapStart && frag && !frag.gap && bufferInfo.nextStart) {
      const nextbufferInfo = this.getFwdBufferInfoAtPos(this.mediaBuffer ? this.mediaBuffer : this.media, bufferInfo.nextStart, playlistType);
      if (nextbufferInfo !== null && bufferInfo.len + nextbufferInfo.len >= maxBufLen) {
        this.log(`buffer full after gaps in "${playlistType}" playlist starting at sn: ${frag.sn}`);
        return null;
      }
    }
    return frag;
  }
  mapToInitFragWhenRequired(frag) {
    if (frag != null && frag.initSegment && !(frag != null && frag.initSegment.data) && !this.bitrateTest) {
      return frag.initSegment;
    }
    return frag;
  }
  getNextPart(partList, frag, targetBufferTime) {
    let nextPart = -1;
    let contiguous = false;
    let independentAttrOmitted = true;
    for (let i2 = 0, len = partList.length; i2 < len; i2++) {
      const part = partList[i2];
      independentAttrOmitted = independentAttrOmitted && !part.independent;
      if (nextPart > -1 && targetBufferTime < part.start) {
        break;
      }
      const loaded = part.loaded;
      if (loaded) {
        nextPart = -1;
      } else if ((contiguous || part.independent || independentAttrOmitted) && part.fragment === frag) {
        nextPart = i2;
      }
      contiguous = loaded;
    }
    return nextPart;
  }
  loadedEndOfParts(partList, targetBufferTime) {
    const lastPart = partList[partList.length - 1];
    return lastPart && targetBufferTime > lastPart.start && lastPart.loaded;
  }
  /*
   This method is used find the best matching first fragment for a live playlist. This fragment is used to calculate the
   "sliding" of the playlist, which is its offset from the start of playback. After sliding we can compute the real
   start and end times for each fragment in the playlist (after which this method will not need to be called).
  */
  getInitialLiveFragment(levelDetails, fragments) {
    const fragPrevious = this.fragPrevious;
    let frag = null;
    if (fragPrevious) {
      if (levelDetails.hasProgramDateTime) {
        this.log(`Live playlist, switching playlist, load frag with same PDT: ${fragPrevious.programDateTime}`);
        frag = findFragmentByPDT(fragments, fragPrevious.endProgramDateTime, this.config.maxFragLookUpTolerance);
      }
      if (!frag) {
        const targetSN = fragPrevious.sn + 1;
        if (targetSN >= levelDetails.startSN && targetSN <= levelDetails.endSN) {
          const fragNext = fragments[targetSN - levelDetails.startSN];
          if (fragPrevious.cc === fragNext.cc) {
            frag = fragNext;
            this.log(`Live playlist, switching playlist, load frag with next SN: ${frag.sn}`);
          }
        }
        if (!frag) {
          frag = findFragWithCC(fragments, fragPrevious.cc);
          if (frag) {
            this.log(`Live playlist, switching playlist, load frag with same CC: ${frag.sn}`);
          }
        }
      }
    } else {
      const liveStart = this.hls.liveSyncPosition;
      if (liveStart !== null) {
        frag = this.getFragmentAtPosition(liveStart, this.bitrateTest ? levelDetails.fragmentEnd : levelDetails.edge, levelDetails);
      }
    }
    return frag;
  }
  /*
  This method finds the best matching fragment given the provided position.
   */
  getFragmentAtPosition(bufferEnd, end, levelDetails) {
    const {
      config
    } = this;
    let {
      fragPrevious
    } = this;
    let {
      fragments,
      endSN
    } = levelDetails;
    const {
      fragmentHint
    } = levelDetails;
    const {
      maxFragLookUpTolerance
    } = config;
    const partList = levelDetails.partList;
    const loadingParts = !!(config.lowLatencyMode && partList != null && partList.length && fragmentHint);
    if (loadingParts && fragmentHint && !this.bitrateTest) {
      fragments = fragments.concat(fragmentHint);
      endSN = fragmentHint.sn;
    }
    let frag;
    if (bufferEnd < end) {
      const lookupTolerance = bufferEnd > end - maxFragLookUpTolerance ? 0 : maxFragLookUpTolerance;
      frag = findFragmentByPTS(fragPrevious, fragments, bufferEnd, lookupTolerance);
    } else {
      frag = fragments[fragments.length - 1];
    }
    if (frag) {
      const curSNIdx = frag.sn - levelDetails.startSN;
      const fragState = this.fragmentTracker.getState(frag);
      if (fragState === FragmentState.OK || fragState === FragmentState.PARTIAL && frag.gap) {
        fragPrevious = frag;
      }
      if (fragPrevious && frag.sn === fragPrevious.sn && (!loadingParts || partList[0].fragment.sn > frag.sn)) {
        const sameLevel = fragPrevious && frag.level === fragPrevious.level;
        if (sameLevel) {
          const nextFrag = fragments[curSNIdx + 1];
          if (frag.sn < endSN && this.fragmentTracker.getState(nextFrag) !== FragmentState.OK) {
            frag = nextFrag;
          } else {
            frag = null;
          }
        }
      }
    }
    return frag;
  }
  synchronizeToLiveEdge(levelDetails) {
    const {
      config,
      media
    } = this;
    if (!media) {
      return;
    }
    const liveSyncPosition = this.hls.liveSyncPosition;
    const currentTime = media.currentTime;
    const start = levelDetails.fragments[0].start;
    const end = levelDetails.edge;
    const withinSlidingWindow = currentTime >= start - config.maxFragLookUpTolerance && currentTime <= end;
    if (liveSyncPosition !== null && media.duration > liveSyncPosition && (currentTime < liveSyncPosition || !withinSlidingWindow)) {
      const maxLatency = config.liveMaxLatencyDuration !== void 0 ? config.liveMaxLatencyDuration : config.liveMaxLatencyDurationCount * levelDetails.targetduration;
      if (!withinSlidingWindow && media.readyState < 4 || currentTime < end - maxLatency) {
        if (!this.loadedmetadata) {
          this.nextLoadPosition = liveSyncPosition;
        }
        if (media.readyState) {
          this.warn(`Playback: ${currentTime.toFixed(3)} is located too far from the end of live sliding playlist: ${end}, reset currentTime to : ${liveSyncPosition.toFixed(3)}`);
          media.currentTime = liveSyncPosition;
        }
      }
    }
  }
  alignPlaylists(details, previousDetails, switchDetails) {
    const length = details.fragments.length;
    if (!length) {
      this.warn(`No fragments in live playlist`);
      return 0;
    }
    const slidingStart = details.fragments[0].start;
    const firstLevelLoad = !previousDetails;
    const aligned = details.alignedSliding && isFiniteNumber(slidingStart);
    if (firstLevelLoad || !aligned && !slidingStart) {
      const {
        fragPrevious
      } = this;
      alignStream(fragPrevious, switchDetails, details);
      const alignedSlidingStart = details.fragments[0].start;
      this.log(`Live playlist sliding: ${alignedSlidingStart.toFixed(2)} start-sn: ${previousDetails ? previousDetails.startSN : "na"}->${details.startSN} prev-sn: ${fragPrevious ? fragPrevious.sn : "na"} fragments: ${length}`);
      return alignedSlidingStart;
    }
    return slidingStart;
  }
  waitForCdnTuneIn(details) {
    const advancePartLimit = 3;
    return details.live && details.canBlockReload && details.partTarget && details.tuneInGoal > Math.max(details.partHoldBack, details.partTarget * advancePartLimit);
  }
  setStartPosition(details, sliding) {
    let startPosition = this.startPosition;
    if (startPosition < sliding) {
      startPosition = -1;
    }
    if (startPosition === -1 || this.lastCurrentTime === -1) {
      const offsetInMultivariantPlaylist = this.startTimeOffset !== null;
      const startTimeOffset = offsetInMultivariantPlaylist ? this.startTimeOffset : details.startTimeOffset;
      if (startTimeOffset !== null && isFiniteNumber(startTimeOffset)) {
        startPosition = sliding + startTimeOffset;
        if (startTimeOffset < 0) {
          startPosition += details.totalduration;
        }
        startPosition = Math.min(Math.max(sliding, startPosition), sliding + details.totalduration);
        this.log(`Start time offset ${startTimeOffset} found in ${offsetInMultivariantPlaylist ? "multivariant" : "media"} playlist, adjust startPosition to ${startPosition}`);
        this.startPosition = startPosition;
      } else if (details.live) {
        startPosition = this.hls.liveSyncPosition || sliding;
      } else {
        this.startPosition = startPosition = 0;
      }
      this.lastCurrentTime = startPosition;
    }
    this.nextLoadPosition = startPosition;
  }
  getLoadPosition() {
    const {
      media
    } = this;
    let pos = 0;
    if (this.loadedmetadata && media) {
      pos = media.currentTime;
    } else if (this.nextLoadPosition) {
      pos = this.nextLoadPosition;
    }
    return pos;
  }
  handleFragLoadAborted(frag, part) {
    if (this.transmuxer && frag.sn !== "initSegment" && frag.stats.aborted) {
      this.warn(`Fragment ${frag.sn}${part ? " part " + part.index : ""} of level ${frag.level} was aborted`);
      this.resetFragmentLoading(frag);
    }
  }
  resetFragmentLoading(frag) {
    if (!this.fragCurrent || !this.fragContextChanged(frag) && this.state !== State.FRAG_LOADING_WAITING_RETRY) {
      this.state = State.IDLE;
    }
  }
  onFragmentOrKeyLoadError(filterType, data) {
    if (data.chunkMeta && !data.frag) {
      const context = this.getCurrentContext(data.chunkMeta);
      if (context) {
        data.frag = context.frag;
      }
    }
    const frag = data.frag;
    if (!frag || frag.type !== filterType || !this.levels) {
      return;
    }
    if (this.fragContextChanged(frag)) {
      var _this$fragCurrent2;
      this.warn(`Frag load error must match current frag to retry ${frag.url} > ${(_this$fragCurrent2 = this.fragCurrent) == null ? void 0 : _this$fragCurrent2.url}`);
      return;
    }
    const gapTagEncountered = data.details === ErrorDetails.FRAG_GAP;
    if (gapTagEncountered) {
      this.fragmentTracker.fragBuffered(frag, true);
    }
    const errorAction = data.errorAction;
    const {
      action,
      retryCount = 0,
      retryConfig
    } = errorAction || {};
    if (errorAction && action === NetworkErrorAction.RetryRequest && retryConfig) {
      this.resetStartWhenNotLoaded(this.levelLastLoaded);
      const delay = getRetryDelay(retryConfig, retryCount);
      this.warn(`Fragment ${frag.sn} of ${filterType} ${frag.level} errored with ${data.details}, retrying loading ${retryCount + 1}/${retryConfig.maxNumRetry} in ${delay}ms`);
      errorAction.resolved = true;
      this.retryDate = self.performance.now() + delay;
      this.state = State.FRAG_LOADING_WAITING_RETRY;
    } else if (retryConfig && errorAction) {
      this.resetFragmentErrors(filterType);
      if (retryCount < retryConfig.maxNumRetry) {
        if (!gapTagEncountered && action !== NetworkErrorAction.RemoveAlternatePermanently) {
          errorAction.resolved = true;
        }
      } else {
        logger.warn(`${data.details} reached or exceeded max retry (${retryCount})`);
        return;
      }
    } else if ((errorAction == null ? void 0 : errorAction.action) === NetworkErrorAction.SendAlternateToPenaltyBox) {
      this.state = State.WAITING_LEVEL;
    } else {
      this.state = State.ERROR;
    }
    this.tickImmediate();
  }
  reduceLengthAndFlushBuffer(data) {
    if (this.state === State.PARSING || this.state === State.PARSED) {
      const frag = data.frag;
      const playlistType = data.parent;
      const bufferedInfo = this.getFwdBufferInfo(this.mediaBuffer, playlistType);
      const buffered = bufferedInfo && bufferedInfo.len > 0.5;
      if (buffered) {
        this.reduceMaxBufferLength(bufferedInfo.len, (frag == null ? void 0 : frag.duration) || 10);
      }
      const flushBuffer = !buffered;
      if (flushBuffer) {
        this.warn(`Buffer full error while media.currentTime is not buffered, flush ${playlistType} buffer`);
      }
      if (frag) {
        this.fragmentTracker.removeFragment(frag);
        this.nextLoadPosition = frag.start;
      }
      this.resetLoadingState();
      return flushBuffer;
    }
    return false;
  }
  resetFragmentErrors(filterType) {
    if (filterType === PlaylistLevelType.AUDIO) {
      this.fragCurrent = null;
    }
    if (!this.loadedmetadata) {
      this.startFragRequested = false;
    }
    if (this.state !== State.STOPPED) {
      this.state = State.IDLE;
    }
  }
  afterBufferFlushed(media, bufferType, playlistType) {
    if (!media) {
      return;
    }
    const bufferedTimeRanges = BufferHelper.getBuffered(media);
    this.fragmentTracker.detectEvictedFragments(bufferType, bufferedTimeRanges, playlistType);
    if (this.state === State.ENDED) {
      this.resetLoadingState();
    }
  }
  resetLoadingState() {
    this.log("Reset loading state");
    this.fragCurrent = null;
    this.fragPrevious = null;
    this.state = State.IDLE;
  }
  resetStartWhenNotLoaded(level) {
    if (!this.loadedmetadata) {
      this.startFragRequested = false;
      const details = level ? level.details : null;
      if (details != null && details.live) {
        this.startPosition = -1;
        this.setStartPosition(details, 0);
        this.resetLoadingState();
      } else {
        this.nextLoadPosition = this.startPosition;
      }
    }
  }
  resetWhenMissingContext(chunkMeta) {
    this.warn(`The loading context changed while buffering fragment ${chunkMeta.sn} of level ${chunkMeta.level}. This chunk will not be buffered.`);
    this.removeUnbufferedFrags();
    this.resetStartWhenNotLoaded(this.levelLastLoaded);
    this.resetLoadingState();
  }
  removeUnbufferedFrags(start = 0) {
    this.fragmentTracker.removeFragmentsInRange(start, Infinity, this.playlistType, false, true);
  }
  updateLevelTiming(frag, part, level, partial) {
    var _this$transmuxer;
    const details = level.details;
    if (!details) {
      this.warn("level.details undefined");
      return;
    }
    const parsed = Object.keys(frag.elementaryStreams).reduce((result, type) => {
      const info = frag.elementaryStreams[type];
      if (info) {
        const parsedDuration = info.endPTS - info.startPTS;
        if (parsedDuration <= 0) {
          this.warn(`Could not parse fragment ${frag.sn} ${type} duration reliably (${parsedDuration})`);
          return result || false;
        }
        const drift = partial ? 0 : updateFragPTSDTS(details, frag, info.startPTS, info.endPTS, info.startDTS, info.endDTS);
        this.hls.trigger(Events.LEVEL_PTS_UPDATED, {
          details,
          level,
          drift,
          type,
          frag,
          start: info.startPTS,
          end: info.endPTS
        });
        return true;
      }
      return result;
    }, false);
    if (!parsed && ((_this$transmuxer = this.transmuxer) == null ? void 0 : _this$transmuxer.error) === null) {
      const error = new Error(`Found no media in fragment ${frag.sn} of level ${frag.level} resetting transmuxer to fallback to playlist timing`);
      if (level.fragmentError === 0) {
        level.fragmentError++;
        frag.gap = true;
        this.fragmentTracker.removeFragment(frag);
        this.fragmentTracker.fragBuffered(frag, true);
      }
      this.warn(error.message);
      this.hls.trigger(Events.ERROR, {
        type: ErrorTypes.MEDIA_ERROR,
        details: ErrorDetails.FRAG_PARSING_ERROR,
        fatal: false,
        error,
        frag,
        reason: `Found no media in msn ${frag.sn} of level "${level.url}"`
      });
      if (!this.hls) {
        return;
      }
      this.resetTransmuxer();
    }
    this.state = State.PARSED;
    this.hls.trigger(Events.FRAG_PARSED, {
      frag,
      part
    });
  }
  resetTransmuxer() {
    if (this.transmuxer) {
      this.transmuxer.destroy();
      this.transmuxer = null;
    }
  }
  recoverWorkerError(data) {
    if (data.event === "demuxerWorker") {
      this.fragmentTracker.removeAllFragments();
      this.resetTransmuxer();
      this.resetStartWhenNotLoaded(this.levelLastLoaded);
      this.resetLoadingState();
    }
  }
  set state(nextState) {
    const previousState = this._state;
    if (previousState !== nextState) {
      this._state = nextState;
      this.log(`${previousState}->${nextState}`);
    }
  }
  get state() {
    return this._state;
  }
}
class ChunkCache {
  constructor() {
    this.chunks = [];
    this.dataLength = 0;
  }
  push(chunk) {
    this.chunks.push(chunk);
    this.dataLength += chunk.length;
  }
  flush() {
    const {
      chunks,
      dataLength
    } = this;
    let result;
    if (!chunks.length) {
      return new Uint8Array(0);
    } else if (chunks.length === 1) {
      result = chunks[0];
    } else {
      result = concatUint8Arrays(chunks, dataLength);
    }
    this.reset();
    return result;
  }
  reset() {
    this.chunks.length = 0;
    this.dataLength = 0;
  }
}
function concatUint8Arrays(chunks, dataLength) {
  const result = new Uint8Array(dataLength);
  let offset = 0;
  for (let i2 = 0; i2 < chunks.length; i2++) {
    const chunk = chunks[i2];
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}
function hasUMDWorker() {
  return typeof __HLS_WORKER_BUNDLE__ === "function";
}
function injectWorker() {
  const blob = new self.Blob([`var exports={};var module={exports:exports};function define(f){f()};define.amd=true;(${__HLS_WORKER_BUNDLE__.toString()})(true);`], {
    type: "text/javascript"
  });
  const objectURL = self.URL.createObjectURL(blob);
  const worker = new self.Worker(objectURL);
  return {
    worker,
    objectURL
  };
}
function loadWorker(path) {
  const scriptURL = new self.URL(path, self.location.href).href;
  const worker = new self.Worker(scriptURL);
  return {
    worker,
    scriptURL
  };
}
function dummyTrack(type = "", inputTimeScale = 9e4) {
  return {
    type,
    id: -1,
    pid: -1,
    inputTimeScale,
    sequenceNumber: -1,
    samples: [],
    dropped: 0
  };
}
class BaseAudioDemuxer {
  constructor() {
    this._audioTrack = void 0;
    this._id3Track = void 0;
    this.frameIndex = 0;
    this.cachedData = null;
    this.basePTS = null;
    this.initPTS = null;
    this.lastPTS = null;
  }
  resetInitSegment(initSegment, audioCodec, videoCodec, trackDuration) {
    this._id3Track = {
      type: "id3",
      id: 3,
      pid: -1,
      inputTimeScale: 9e4,
      sequenceNumber: 0,
      samples: [],
      dropped: 0
    };
  }
  resetTimeStamp(deaultTimestamp) {
    this.initPTS = deaultTimestamp;
    this.resetContiguity();
  }
  resetContiguity() {
    this.basePTS = null;
    this.lastPTS = null;
    this.frameIndex = 0;
  }
  canParse(data, offset) {
    return false;
  }
  appendFrame(track, data, offset) {
  }
  // feed incoming data to the front of the parsing pipeline
  demux(data, timeOffset) {
    if (this.cachedData) {
      data = appendUint8Array(this.cachedData, data);
      this.cachedData = null;
    }
    let id3Data = getID3Data(data, 0);
    let offset = id3Data ? id3Data.length : 0;
    let lastDataIndex;
    const track = this._audioTrack;
    const id3Track = this._id3Track;
    const timestamp = id3Data ? getTimeStamp(id3Data) : void 0;
    const length = data.length;
    if (this.basePTS === null || this.frameIndex === 0 && isFiniteNumber(timestamp)) {
      this.basePTS = initPTSFn(timestamp, timeOffset, this.initPTS);
      this.lastPTS = this.basePTS;
    }
    if (this.lastPTS === null) {
      this.lastPTS = this.basePTS;
    }
    if (id3Data && id3Data.length > 0) {
      id3Track.samples.push({
        pts: this.lastPTS,
        dts: this.lastPTS,
        data: id3Data,
        type: MetadataSchema.audioId3,
        duration: Number.POSITIVE_INFINITY
      });
    }
    while (offset < length) {
      if (this.canParse(data, offset)) {
        const frame = this.appendFrame(track, data, offset);
        if (frame) {
          this.frameIndex++;
          this.lastPTS = frame.sample.pts;
          offset += frame.length;
          lastDataIndex = offset;
        } else {
          offset = length;
        }
      } else if (canParse$2(data, offset)) {
        id3Data = getID3Data(data, offset);
        id3Track.samples.push({
          pts: this.lastPTS,
          dts: this.lastPTS,
          data: id3Data,
          type: MetadataSchema.audioId3,
          duration: Number.POSITIVE_INFINITY
        });
        offset += id3Data.length;
        lastDataIndex = offset;
      } else {
        offset++;
      }
      if (offset === length && lastDataIndex !== length) {
        const partialData = sliceUint8(data, lastDataIndex);
        if (this.cachedData) {
          this.cachedData = appendUint8Array(this.cachedData, partialData);
        } else {
          this.cachedData = partialData;
        }
      }
    }
    return {
      audioTrack: track,
      videoTrack: dummyTrack(),
      id3Track,
      textTrack: dummyTrack()
    };
  }
  demuxSampleAes(data, keyData, timeOffset) {
    return Promise.reject(new Error(`[${this}] This demuxer does not support Sample-AES decryption`));
  }
  flush(timeOffset) {
    const cachedData = this.cachedData;
    if (cachedData) {
      this.cachedData = null;
      this.demux(cachedData, 0);
    }
    return {
      audioTrack: this._audioTrack,
      videoTrack: dummyTrack(),
      id3Track: this._id3Track,
      textTrack: dummyTrack()
    };
  }
  destroy() {
  }
}
const initPTSFn = (timestamp, timeOffset, initPTS) => {
  if (isFiniteNumber(timestamp)) {
    return timestamp * 90;
  }
  const init90kHz = initPTS ? initPTS.baseTime * 9e4 / initPTS.timescale : 0;
  return timeOffset * 9e4 + init90kHz;
};
function getAudioConfig(observer, data, offset, audioCodec) {
  let adtsObjectType;
  let adtsExtensionSamplingIndex;
  let adtsChannelConfig;
  let config;
  const userAgent = navigator.userAgent.toLowerCase();
  const manifestCodec = audioCodec;
  const adtsSamplingRates = [96e3, 88200, 64e3, 48e3, 44100, 32e3, 24e3, 22050, 16e3, 12e3, 11025, 8e3, 7350];
  adtsObjectType = ((data[offset + 2] & 192) >>> 6) + 1;
  const adtsSamplingIndex = (data[offset + 2] & 60) >>> 2;
  if (adtsSamplingIndex > adtsSamplingRates.length - 1) {
    const error = new Error(`invalid ADTS sampling index:${adtsSamplingIndex}`);
    observer.emit(Events.ERROR, Events.ERROR, {
      type: ErrorTypes.MEDIA_ERROR,
      details: ErrorDetails.FRAG_PARSING_ERROR,
      fatal: true,
      error,
      reason: error.message
    });
    return;
  }
  adtsChannelConfig = (data[offset + 2] & 1) << 2;
  adtsChannelConfig |= (data[offset + 3] & 192) >>> 6;
  logger.log(`manifest codec:${audioCodec}, ADTS type:${adtsObjectType}, samplingIndex:${adtsSamplingIndex}`);
  if (/firefox/i.test(userAgent)) {
    if (adtsSamplingIndex >= 6) {
      adtsObjectType = 5;
      config = new Array(4);
      adtsExtensionSamplingIndex = adtsSamplingIndex - 3;
    } else {
      adtsObjectType = 2;
      config = new Array(2);
      adtsExtensionSamplingIndex = adtsSamplingIndex;
    }
  } else if (userAgent.indexOf("android") !== -1) {
    adtsObjectType = 2;
    config = new Array(2);
    adtsExtensionSamplingIndex = adtsSamplingIndex;
  } else {
    adtsObjectType = 5;
    config = new Array(4);
    if (audioCodec && (audioCodec.indexOf("mp4a.40.29") !== -1 || audioCodec.indexOf("mp4a.40.5") !== -1) || !audioCodec && adtsSamplingIndex >= 6) {
      adtsExtensionSamplingIndex = adtsSamplingIndex - 3;
    } else {
      if (audioCodec && audioCodec.indexOf("mp4a.40.2") !== -1 && (adtsSamplingIndex >= 6 && adtsChannelConfig === 1 || /vivaldi/i.test(userAgent)) || !audioCodec && adtsChannelConfig === 1) {
        adtsObjectType = 2;
        config = new Array(2);
      }
      adtsExtensionSamplingIndex = adtsSamplingIndex;
    }
  }
  config[0] = adtsObjectType << 3;
  config[0] |= (adtsSamplingIndex & 14) >> 1;
  config[1] |= (adtsSamplingIndex & 1) << 7;
  config[1] |= adtsChannelConfig << 3;
  if (adtsObjectType === 5) {
    config[1] |= (adtsExtensionSamplingIndex & 14) >> 1;
    config[2] = (adtsExtensionSamplingIndex & 1) << 7;
    config[2] |= 2 << 2;
    config[3] = 0;
  }
  return {
    config,
    samplerate: adtsSamplingRates[adtsSamplingIndex],
    channelCount: adtsChannelConfig,
    codec: "mp4a.40." + adtsObjectType,
    manifestCodec
  };
}
function isHeaderPattern$1(data, offset) {
  return data[offset] === 255 && (data[offset + 1] & 246) === 240;
}
function getHeaderLength(data, offset) {
  return data[offset + 1] & 1 ? 7 : 9;
}
function getFullFrameLength(data, offset) {
  return (data[offset + 3] & 3) << 11 | data[offset + 4] << 3 | (data[offset + 5] & 224) >>> 5;
}
function canGetFrameLength(data, offset) {
  return offset + 5 < data.length;
}
function isHeader$1(data, offset) {
  return offset + 1 < data.length && isHeaderPattern$1(data, offset);
}
function canParse$1(data, offset) {
  return canGetFrameLength(data, offset) && isHeaderPattern$1(data, offset) && getFullFrameLength(data, offset) <= data.length - offset;
}
function probe$1(data, offset) {
  if (isHeader$1(data, offset)) {
    const headerLength = getHeaderLength(data, offset);
    if (offset + headerLength >= data.length) {
      return false;
    }
    const frameLength = getFullFrameLength(data, offset);
    if (frameLength <= headerLength) {
      return false;
    }
    const newOffset = offset + frameLength;
    return newOffset === data.length || isHeader$1(data, newOffset);
  }
  return false;
}
function initTrackConfig(track, observer, data, offset, audioCodec) {
  if (!track.samplerate) {
    const config = getAudioConfig(observer, data, offset, audioCodec);
    if (!config) {
      return;
    }
    track.config = config.config;
    track.samplerate = config.samplerate;
    track.channelCount = config.channelCount;
    track.codec = config.codec;
    track.manifestCodec = config.manifestCodec;
    logger.log(`parsed codec:${track.codec}, rate:${config.samplerate}, channels:${config.channelCount}`);
  }
}
function getFrameDuration(samplerate) {
  return 1024 * 9e4 / samplerate;
}
function parseFrameHeader(data, offset) {
  const headerLength = getHeaderLength(data, offset);
  if (offset + headerLength <= data.length) {
    const frameLength = getFullFrameLength(data, offset) - headerLength;
    if (frameLength > 0) {
      return {
        headerLength,
        frameLength
      };
    }
  }
}
function appendFrame$2(track, data, offset, pts, frameIndex) {
  const frameDuration = getFrameDuration(track.samplerate);
  const stamp = pts + frameIndex * frameDuration;
  const header = parseFrameHeader(data, offset);
  let unit;
  if (header) {
    const {
      frameLength,
      headerLength
    } = header;
    const _length = headerLength + frameLength;
    const missing = Math.max(0, offset + _length - data.length);
    if (missing) {
      unit = new Uint8Array(_length - headerLength);
      unit.set(data.subarray(offset + headerLength, data.length), 0);
    } else {
      unit = data.subarray(offset + headerLength, offset + _length);
    }
    const _sample = {
      unit,
      pts: stamp
    };
    if (!missing) {
      track.samples.push(_sample);
    }
    return {
      sample: _sample,
      length: _length,
      missing
    };
  }
  const length = data.length - offset;
  unit = new Uint8Array(length);
  unit.set(data.subarray(offset, data.length), 0);
  const sample = {
    unit,
    pts: stamp
  };
  return {
    sample,
    length,
    missing: -1
  };
}
let chromeVersion$1 = null;
const BitratesMap = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160];
const SamplingRateMap = [44100, 48e3, 32e3, 22050, 24e3, 16e3, 11025, 12e3, 8e3];
const SamplesCoefficients = [
  // MPEG 2.5
  [
    0,
    // Reserved
    72,
    // Layer3
    144,
    // Layer2
    12
    // Layer1
  ],
  // Reserved
  [
    0,
    // Reserved
    0,
    // Layer3
    0,
    // Layer2
    0
    // Layer1
  ],
  // MPEG 2
  [
    0,
    // Reserved
    72,
    // Layer3
    144,
    // Layer2
    12
    // Layer1
  ],
  // MPEG 1
  [
    0,
    // Reserved
    144,
    // Layer3
    144,
    // Layer2
    12
    // Layer1
  ]
];
const BytesInSlot = [
  0,
  // Reserved
  1,
  // Layer3
  1,
  // Layer2
  4
  // Layer1
];
function appendFrame$1(track, data, offset, pts, frameIndex) {
  if (offset + 24 > data.length) {
    return;
  }
  const header = parseHeader(data, offset);
  if (header && offset + header.frameLength <= data.length) {
    const frameDuration = header.samplesPerFrame * 9e4 / header.sampleRate;
    const stamp = pts + frameIndex * frameDuration;
    const sample = {
      unit: data.subarray(offset, offset + header.frameLength),
      pts: stamp,
      dts: stamp
    };
    track.config = [];
    track.channelCount = header.channelCount;
    track.samplerate = header.sampleRate;
    track.samples.push(sample);
    return {
      sample,
      length: header.frameLength,
      missing: 0
    };
  }
}
function parseHeader(data, offset) {
  const mpegVersion = data[offset + 1] >> 3 & 3;
  const mpegLayer = data[offset + 1] >> 1 & 3;
  const bitRateIndex = data[offset + 2] >> 4 & 15;
  const sampleRateIndex = data[offset + 2] >> 2 & 3;
  if (mpegVersion !== 1 && bitRateIndex !== 0 && bitRateIndex !== 15 && sampleRateIndex !== 3) {
    const paddingBit = data[offset + 2] >> 1 & 1;
    const channelMode = data[offset + 3] >> 6;
    const columnInBitrates = mpegVersion === 3 ? 3 - mpegLayer : mpegLayer === 3 ? 3 : 4;
    const bitRate = BitratesMap[columnInBitrates * 14 + bitRateIndex - 1] * 1e3;
    const columnInSampleRates = mpegVersion === 3 ? 0 : mpegVersion === 2 ? 1 : 2;
    const sampleRate = SamplingRateMap[columnInSampleRates * 3 + sampleRateIndex];
    const channelCount = channelMode === 3 ? 1 : 2;
    const sampleCoefficient = SamplesCoefficients[mpegVersion][mpegLayer];
    const bytesInSlot = BytesInSlot[mpegLayer];
    const samplesPerFrame = sampleCoefficient * 8 * bytesInSlot;
    const frameLength = Math.floor(sampleCoefficient * bitRate / sampleRate + paddingBit) * bytesInSlot;
    if (chromeVersion$1 === null) {
      const userAgent = navigator.userAgent || "";
      const result = userAgent.match(/Chrome\/(\d+)/i);
      chromeVersion$1 = result ? parseInt(result[1]) : 0;
    }
    const needChromeFix = !!chromeVersion$1 && chromeVersion$1 <= 87;
    if (needChromeFix && mpegLayer === 2 && bitRate >= 224e3 && channelMode === 0) {
      data[offset + 3] = data[offset + 3] | 128;
    }
    return {
      sampleRate,
      channelCount,
      frameLength,
      samplesPerFrame
    };
  }
}
function isHeaderPattern(data, offset) {
  return data[offset] === 255 && (data[offset + 1] & 224) === 224 && (data[offset + 1] & 6) !== 0;
}
function isHeader(data, offset) {
  return offset + 1 < data.length && isHeaderPattern(data, offset);
}
function canParse(data, offset) {
  const headerSize = 4;
  return isHeaderPattern(data, offset) && headerSize <= data.length - offset;
}
function probe(data, offset) {
  if (offset + 1 < data.length && isHeaderPattern(data, offset)) {
    const headerLength = 4;
    const header = parseHeader(data, offset);
    let frameLength = headerLength;
    if (header != null && header.frameLength) {
      frameLength = header.frameLength;
    }
    const newOffset = offset + frameLength;
    return newOffset === data.length || isHeader(data, newOffset);
  }
  return false;
}
class AACDemuxer extends BaseAudioDemuxer {
  constructor(observer, config) {
    super();
    this.observer = void 0;
    this.config = void 0;
    this.observer = observer;
    this.config = config;
  }
  resetInitSegment(initSegment, audioCodec, videoCodec, trackDuration) {
    super.resetInitSegment(initSegment, audioCodec, videoCodec, trackDuration);
    this._audioTrack = {
      container: "audio/adts",
      type: "audio",
      id: 2,
      pid: -1,
      sequenceNumber: 0,
      segmentCodec: "aac",
      samples: [],
      manifestCodec: audioCodec,
      duration: trackDuration,
      inputTimeScale: 9e4,
      dropped: 0
    };
  }
  // Source for probe info - https://wiki.multimedia.cx/index.php?title=ADTS
  static probe(data) {
    if (!data) {
      return false;
    }
    const id3Data = getID3Data(data, 0);
    let offset = (id3Data == null ? void 0 : id3Data.length) || 0;
    if (probe(data, offset)) {
      return false;
    }
    for (let length = data.length; offset < length; offset++) {
      if (probe$1(data, offset)) {
        logger.log("ADTS sync word found !");
        return true;
      }
    }
    return false;
  }
  canParse(data, offset) {
    return canParse$1(data, offset);
  }
  appendFrame(track, data, offset) {
    initTrackConfig(track, this.observer, data, offset, track.manifestCodec);
    const frame = appendFrame$2(track, data, offset, this.basePTS, this.frameIndex);
    if (frame && frame.missing === 0) {
      return frame;
    }
  }
}
const emsgSchemePattern = /\/emsg[-/]ID3/i;
class MP4Demuxer {
  constructor(observer, config) {
    this.remainderData = null;
    this.timeOffset = 0;
    this.config = void 0;
    this.videoTrack = void 0;
    this.audioTrack = void 0;
    this.id3Track = void 0;
    this.txtTrack = void 0;
    this.config = config;
  }
  resetTimeStamp() {
  }
  resetInitSegment(initSegment, audioCodec, videoCodec, trackDuration) {
    const videoTrack = this.videoTrack = dummyTrack("video", 1);
    const audioTrack = this.audioTrack = dummyTrack("audio", 1);
    const captionTrack = this.txtTrack = dummyTrack("text", 1);
    this.id3Track = dummyTrack("id3", 1);
    this.timeOffset = 0;
    if (!(initSegment != null && initSegment.byteLength)) {
      return;
    }
    const initData = parseInitSegment(initSegment);
    if (initData.video) {
      const {
        id,
        timescale,
        codec
      } = initData.video;
      videoTrack.id = id;
      videoTrack.timescale = captionTrack.timescale = timescale;
      videoTrack.codec = codec;
    }
    if (initData.audio) {
      const {
        id,
        timescale,
        codec
      } = initData.audio;
      audioTrack.id = id;
      audioTrack.timescale = timescale;
      audioTrack.codec = codec;
    }
    captionTrack.id = RemuxerTrackIdConfig.text;
    videoTrack.sampleDuration = 0;
    videoTrack.duration = audioTrack.duration = trackDuration;
  }
  resetContiguity() {
    this.remainderData = null;
  }
  static probe(data) {
    return hasMoofData(data);
  }
  demux(data, timeOffset) {
    this.timeOffset = timeOffset;
    let videoSamples = data;
    const videoTrack = this.videoTrack;
    const textTrack = this.txtTrack;
    if (this.config.progressive) {
      if (this.remainderData) {
        videoSamples = appendUint8Array(this.remainderData, data);
      }
      const segmentedData = segmentValidRange(videoSamples);
      this.remainderData = segmentedData.remainder;
      videoTrack.samples = segmentedData.valid || new Uint8Array();
    } else {
      videoTrack.samples = videoSamples;
    }
    const id3Track = this.extractID3Track(videoTrack, timeOffset);
    textTrack.samples = parseSamples(timeOffset, videoTrack);
    return {
      videoTrack,
      audioTrack: this.audioTrack,
      id3Track,
      textTrack: this.txtTrack
    };
  }
  flush() {
    const timeOffset = this.timeOffset;
    const videoTrack = this.videoTrack;
    const textTrack = this.txtTrack;
    videoTrack.samples = this.remainderData || new Uint8Array();
    this.remainderData = null;
    const id3Track = this.extractID3Track(videoTrack, this.timeOffset);
    textTrack.samples = parseSamples(timeOffset, videoTrack);
    return {
      videoTrack,
      audioTrack: dummyTrack(),
      id3Track,
      textTrack: dummyTrack()
    };
  }
  extractID3Track(videoTrack, timeOffset) {
    const id3Track = this.id3Track;
    if (videoTrack.samples.length) {
      const emsgs = findBox(videoTrack.samples, ["emsg"]);
      if (emsgs) {
        emsgs.forEach((data) => {
          const emsgInfo = parseEmsg(data);
          if (emsgSchemePattern.test(emsgInfo.schemeIdUri)) {
            const pts = isFiniteNumber(emsgInfo.presentationTime) ? emsgInfo.presentationTime / emsgInfo.timeScale : timeOffset + emsgInfo.presentationTimeDelta / emsgInfo.timeScale;
            let duration = emsgInfo.eventDuration === 4294967295 ? Number.POSITIVE_INFINITY : emsgInfo.eventDuration / emsgInfo.timeScale;
            if (duration <= 1e-3) {
              duration = Number.POSITIVE_INFINITY;
            }
            const payload = emsgInfo.payload;
            id3Track.samples.push({
              data: payload,
              len: payload.byteLength,
              dts: pts,
              pts,
              type: MetadataSchema.emsg,
              duration
            });
          }
        });
      }
    }
    return id3Track;
  }
  demuxSampleAes(data, keyData, timeOffset) {
    return Promise.reject(new Error("The MP4 demuxer does not support SAMPLE-AES decryption"));
  }
  destroy() {
  }
}
const getAudioBSID = (data, offset) => {
  let bsid = 0;
  let numBits = 5;
  offset += numBits;
  const temp = new Uint32Array(1);
  const mask = new Uint32Array(1);
  const byte = new Uint8Array(1);
  while (numBits > 0) {
    byte[0] = data[offset];
    const bits = Math.min(numBits, 8);
    const shift = 8 - bits;
    mask[0] = 4278190080 >>> 24 + shift << shift;
    temp[0] = (byte[0] & mask[0]) >> shift;
    bsid = !bsid ? temp[0] : bsid << bits | temp[0];
    offset += 1;
    numBits -= bits;
  }
  return bsid;
};
class AC3Demuxer extends BaseAudioDemuxer {
  constructor(observer) {
    super();
    this.observer = void 0;
    this.observer = observer;
  }
  resetInitSegment(initSegment, audioCodec, videoCodec, trackDuration) {
    super.resetInitSegment(initSegment, audioCodec, videoCodec, trackDuration);
    this._audioTrack = {
      container: "audio/ac-3",
      type: "audio",
      id: 2,
      pid: -1,
      sequenceNumber: 0,
      segmentCodec: "ac3",
      samples: [],
      manifestCodec: audioCodec,
      duration: trackDuration,
      inputTimeScale: 9e4,
      dropped: 0
    };
  }
  canParse(data, offset) {
    return offset + 64 < data.length;
  }
  appendFrame(track, data, offset) {
    const frameLength = appendFrame(track, data, offset, this.basePTS, this.frameIndex);
    if (frameLength !== -1) {
      const sample = track.samples[track.samples.length - 1];
      return {
        sample,
        length: frameLength,
        missing: 0
      };
    }
  }
  static probe(data) {
    if (!data) {
      return false;
    }
    const id3Data = getID3Data(data, 0);
    if (!id3Data) {
      return false;
    }
    const offset = id3Data.length;
    if (data[offset] === 11 && data[offset + 1] === 119 && getTimeStamp(id3Data) !== void 0 && // check the bsid to confirm ac-3
    getAudioBSID(data, offset) < 16) {
      return true;
    }
    return false;
  }
}
function appendFrame(track, data, start, pts, frameIndex) {
  if (start + 8 > data.length) {
    return -1;
  }
  if (data[start] !== 11 || data[start + 1] !== 119) {
    return -1;
  }
  const samplingRateCode = data[start + 4] >> 6;
  if (samplingRateCode >= 3) {
    return -1;
  }
  const samplingRateMap = [48e3, 44100, 32e3];
  const sampleRate = samplingRateMap[samplingRateCode];
  const frameSizeCode = data[start + 4] & 63;
  const frameSizeMap = [64, 69, 96, 64, 70, 96, 80, 87, 120, 80, 88, 120, 96, 104, 144, 96, 105, 144, 112, 121, 168, 112, 122, 168, 128, 139, 192, 128, 140, 192, 160, 174, 240, 160, 175, 240, 192, 208, 288, 192, 209, 288, 224, 243, 336, 224, 244, 336, 256, 278, 384, 256, 279, 384, 320, 348, 480, 320, 349, 480, 384, 417, 576, 384, 418, 576, 448, 487, 672, 448, 488, 672, 512, 557, 768, 512, 558, 768, 640, 696, 960, 640, 697, 960, 768, 835, 1152, 768, 836, 1152, 896, 975, 1344, 896, 976, 1344, 1024, 1114, 1536, 1024, 1115, 1536, 1152, 1253, 1728, 1152, 1254, 1728, 1280, 1393, 1920, 1280, 1394, 1920];
  const frameLength = frameSizeMap[frameSizeCode * 3 + samplingRateCode] * 2;
  if (start + frameLength > data.length) {
    return -1;
  }
  const channelMode = data[start + 6] >> 5;
  let skipCount = 0;
  if (channelMode === 2) {
    skipCount += 2;
  } else {
    if (channelMode & 1 && channelMode !== 1) {
      skipCount += 2;
    }
    if (channelMode & 4) {
      skipCount += 2;
    }
  }
  const lfeon = (data[start + 6] << 8 | data[start + 7]) >> 12 - skipCount & 1;
  const channelsMap = [2, 1, 2, 3, 3, 4, 4, 5];
  const channelCount = channelsMap[channelMode] + lfeon;
  const bsid = data[start + 5] >> 3;
  const bsmod = data[start + 5] & 7;
  const config = new Uint8Array([samplingRateCode << 6 | bsid << 1 | bsmod >> 2, (bsmod & 3) << 6 | channelMode << 3 | lfeon << 2 | frameSizeCode >> 4, frameSizeCode << 4 & 224]);
  const frameDuration = 1536 / sampleRate * 9e4;
  const stamp = pts + frameIndex * frameDuration;
  const unit = data.subarray(start, start + frameLength);
  track.config = config;
  track.channelCount = channelCount;
  track.samplerate = sampleRate;
  track.samples.push({
    unit,
    pts: stamp
  });
  return frameLength;
}
class BaseVideoParser {
  constructor() {
    this.VideoSample = null;
  }
  createVideoSample(key, pts, dts, debug) {
    return {
      key,
      frame: false,
      pts,
      dts,
      units: [],
      debug,
      length: 0
    };
  }
  getLastNalUnit(samples) {
    var _VideoSample;
    let VideoSample = this.VideoSample;
    let lastUnit;
    if (!VideoSample || VideoSample.units.length === 0) {
      VideoSample = samples[samples.length - 1];
    }
    if ((_VideoSample = VideoSample) != null && _VideoSample.units) {
      const units = VideoSample.units;
      lastUnit = units[units.length - 1];
    }
    return lastUnit;
  }
  pushAccessUnit(VideoSample, videoTrack) {
    if (VideoSample.units.length && VideoSample.frame) {
      if (VideoSample.pts === void 0) {
        const samples = videoTrack.samples;
        const nbSamples = samples.length;
        if (nbSamples) {
          const lastSample = samples[nbSamples - 1];
          VideoSample.pts = lastSample.pts;
          VideoSample.dts = lastSample.dts;
        } else {
          videoTrack.dropped++;
          return;
        }
      }
      videoTrack.samples.push(VideoSample);
    }
    if (VideoSample.debug.length) {
      logger.log(VideoSample.pts + "/" + VideoSample.dts + ":" + VideoSample.debug);
    }
  }
}
class ExpGolomb {
  constructor(data) {
    this.data = void 0;
    this.bytesAvailable = void 0;
    this.word = void 0;
    this.bitsAvailable = void 0;
    this.data = data;
    this.bytesAvailable = data.byteLength;
    this.word = 0;
    this.bitsAvailable = 0;
  }
  // ():void
  loadWord() {
    const data = this.data;
    const bytesAvailable = this.bytesAvailable;
    const position = data.byteLength - bytesAvailable;
    const workingBytes = new Uint8Array(4);
    const availableBytes = Math.min(4, bytesAvailable);
    if (availableBytes === 0) {
      throw new Error("no bytes available");
    }
    workingBytes.set(data.subarray(position, position + availableBytes));
    this.word = new DataView(workingBytes.buffer).getUint32(0);
    this.bitsAvailable = availableBytes * 8;
    this.bytesAvailable -= availableBytes;
  }
  // (count:int):void
  skipBits(count) {
    let skipBytes;
    count = Math.min(count, this.bytesAvailable * 8 + this.bitsAvailable);
    if (this.bitsAvailable > count) {
      this.word <<= count;
      this.bitsAvailable -= count;
    } else {
      count -= this.bitsAvailable;
      skipBytes = count >> 3;
      count -= skipBytes << 3;
      this.bytesAvailable -= skipBytes;
      this.loadWord();
      this.word <<= count;
      this.bitsAvailable -= count;
    }
  }
  // (size:int):uint
  readBits(size) {
    let bits = Math.min(this.bitsAvailable, size);
    const valu = this.word >>> 32 - bits;
    if (size > 32) {
      logger.error("Cannot read more than 32 bits at a time");
    }
    this.bitsAvailable -= bits;
    if (this.bitsAvailable > 0) {
      this.word <<= bits;
    } else if (this.bytesAvailable > 0) {
      this.loadWord();
    } else {
      throw new Error("no bits available");
    }
    bits = size - bits;
    if (bits > 0 && this.bitsAvailable) {
      return valu << bits | this.readBits(bits);
    } else {
      return valu;
    }
  }
  // ():uint
  skipLZ() {
    let leadingZeroCount;
    for (leadingZeroCount = 0; leadingZeroCount < this.bitsAvailable; ++leadingZeroCount) {
      if ((this.word & 2147483648 >>> leadingZeroCount) !== 0) {
        this.word <<= leadingZeroCount;
        this.bitsAvailable -= leadingZeroCount;
        return leadingZeroCount;
      }
    }
    this.loadWord();
    return leadingZeroCount + this.skipLZ();
  }
  // ():void
  skipUEG() {
    this.skipBits(1 + this.skipLZ());
  }
  // ():void
  skipEG() {
    this.skipBits(1 + this.skipLZ());
  }
  // ():uint
  readUEG() {
    const clz = this.skipLZ();
    return this.readBits(clz + 1) - 1;
  }
  // ():int
  readEG() {
    const valu = this.readUEG();
    if (1 & valu) {
      return 1 + valu >>> 1;
    } else {
      return -1 * (valu >>> 1);
    }
  }
  // Some convenience functions
  // :Boolean
  readBoolean() {
    return this.readBits(1) === 1;
  }
  // ():int
  readUByte() {
    return this.readBits(8);
  }
  // ():int
  readUShort() {
    return this.readBits(16);
  }
  // ():int
  readUInt() {
    return this.readBits(32);
  }
  /**
   * Advance the ExpGolomb decoder past a scaling list. The scaling
   * list is optionally transmitted as part of a sequence parameter
   * set and is not relevant to transmuxing.
   * @param count the number of entries in this scaling list
   * @see Recommendation ITU-T H.264, Section 7.3.2.1.1.1
   */
  skipScalingList(count) {
    let lastScale = 8;
    let nextScale = 8;
    let deltaScale;
    for (let j = 0; j < count; j++) {
      if (nextScale !== 0) {
        deltaScale = this.readEG();
        nextScale = (lastScale + deltaScale + 256) % 256;
      }
      lastScale = nextScale === 0 ? lastScale : nextScale;
    }
  }
  /**
   * Read a sequence parameter set and return some interesting video
   * properties. A sequence parameter set is the H264 metadata that
   * describes the properties of upcoming video frames.
   * @returns an object with configuration parsed from the
   * sequence parameter set, including the dimensions of the
   * associated video frames.
   */
  readSPS() {
    let frameCropLeftOffset = 0;
    let frameCropRightOffset = 0;
    let frameCropTopOffset = 0;
    let frameCropBottomOffset = 0;
    let numRefFramesInPicOrderCntCycle;
    let scalingListCount;
    let i2;
    const readUByte = this.readUByte.bind(this);
    const readBits = this.readBits.bind(this);
    const readUEG = this.readUEG.bind(this);
    const readBoolean = this.readBoolean.bind(this);
    const skipBits = this.skipBits.bind(this);
    const skipEG = this.skipEG.bind(this);
    const skipUEG = this.skipUEG.bind(this);
    const skipScalingList = this.skipScalingList.bind(this);
    readUByte();
    const profileIdc = readUByte();
    readBits(5);
    skipBits(3);
    readUByte();
    skipUEG();
    if (profileIdc === 100 || profileIdc === 110 || profileIdc === 122 || profileIdc === 244 || profileIdc === 44 || profileIdc === 83 || profileIdc === 86 || profileIdc === 118 || profileIdc === 128) {
      const chromaFormatIdc = readUEG();
      if (chromaFormatIdc === 3) {
        skipBits(1);
      }
      skipUEG();
      skipUEG();
      skipBits(1);
      if (readBoolean()) {
        scalingListCount = chromaFormatIdc !== 3 ? 8 : 12;
        for (i2 = 0; i2 < scalingListCount; i2++) {
          if (readBoolean()) {
            if (i2 < 6) {
              skipScalingList(16);
            } else {
              skipScalingList(64);
            }
          }
        }
      }
    }
    skipUEG();
    const picOrderCntType = readUEG();
    if (picOrderCntType === 0) {
      readUEG();
    } else if (picOrderCntType === 1) {
      skipBits(1);
      skipEG();
      skipEG();
      numRefFramesInPicOrderCntCycle = readUEG();
      for (i2 = 0; i2 < numRefFramesInPicOrderCntCycle; i2++) {
        skipEG();
      }
    }
    skipUEG();
    skipBits(1);
    const picWidthInMbsMinus1 = readUEG();
    const picHeightInMapUnitsMinus1 = readUEG();
    const frameMbsOnlyFlag = readBits(1);
    if (frameMbsOnlyFlag === 0) {
      skipBits(1);
    }
    skipBits(1);
    if (readBoolean()) {
      frameCropLeftOffset = readUEG();
      frameCropRightOffset = readUEG();
      frameCropTopOffset = readUEG();
      frameCropBottomOffset = readUEG();
    }
    let pixelRatio = [1, 1];
    if (readBoolean()) {
      if (readBoolean()) {
        const aspectRatioIdc = readUByte();
        switch (aspectRatioIdc) {
          case 1:
            pixelRatio = [1, 1];
            break;
          case 2:
            pixelRatio = [12, 11];
            break;
          case 3:
            pixelRatio = [10, 11];
            break;
          case 4:
            pixelRatio = [16, 11];
            break;
          case 5:
            pixelRatio = [40, 33];
            break;
          case 6:
            pixelRatio = [24, 11];
            break;
          case 7:
            pixelRatio = [20, 11];
            break;
          case 8:
            pixelRatio = [32, 11];
            break;
          case 9:
            pixelRatio = [80, 33];
            break;
          case 10:
            pixelRatio = [18, 11];
            break;
          case 11:
            pixelRatio = [15, 11];
            break;
          case 12:
            pixelRatio = [64, 33];
            break;
          case 13:
            pixelRatio = [160, 99];
            break;
          case 14:
            pixelRatio = [4, 3];
            break;
          case 15:
            pixelRatio = [3, 2];
            break;
          case 16:
            pixelRatio = [2, 1];
            break;
          case 255: {
            pixelRatio = [readUByte() << 8 | readUByte(), readUByte() << 8 | readUByte()];
            break;
          }
        }
      }
    }
    return {
      width: Math.ceil((picWidthInMbsMinus1 + 1) * 16 - frameCropLeftOffset * 2 - frameCropRightOffset * 2),
      height: (2 - frameMbsOnlyFlag) * (picHeightInMapUnitsMinus1 + 1) * 16 - (frameMbsOnlyFlag ? 2 : 4) * (frameCropTopOffset + frameCropBottomOffset),
      pixelRatio
    };
  }
  readSliceType() {
    this.readUByte();
    this.readUEG();
    return this.readUEG();
  }
}
class AvcVideoParser extends BaseVideoParser {
  parseAVCPES(track, textTrack, pes, last, duration) {
    const units = this.parseAVCNALu(track, pes.data);
    let VideoSample = this.VideoSample;
    let push2;
    let spsfound = false;
    pes.data = null;
    if (VideoSample && units.length && !track.audFound) {
      this.pushAccessUnit(VideoSample, track);
      VideoSample = this.VideoSample = this.createVideoSample(false, pes.pts, pes.dts, "");
    }
    units.forEach((unit) => {
      var _VideoSample2;
      switch (unit.type) {
        // NDR
        case 1: {
          let iskey = false;
          push2 = true;
          const data = unit.data;
          if (spsfound && data.length > 4) {
            const sliceType = new ExpGolomb(data).readSliceType();
            if (sliceType === 2 || sliceType === 4 || sliceType === 7 || sliceType === 9) {
              iskey = true;
            }
          }
          if (iskey) {
            var _VideoSample;
            if ((_VideoSample = VideoSample) != null && _VideoSample.frame && !VideoSample.key) {
              this.pushAccessUnit(VideoSample, track);
              VideoSample = this.VideoSample = null;
            }
          }
          if (!VideoSample) {
            VideoSample = this.VideoSample = this.createVideoSample(true, pes.pts, pes.dts, "");
          }
          VideoSample.frame = true;
          VideoSample.key = iskey;
          break;
        }
        case 5:
          push2 = true;
          if ((_VideoSample2 = VideoSample) != null && _VideoSample2.frame && !VideoSample.key) {
            this.pushAccessUnit(VideoSample, track);
            VideoSample = this.VideoSample = null;
          }
          if (!VideoSample) {
            VideoSample = this.VideoSample = this.createVideoSample(true, pes.pts, pes.dts, "");
          }
          VideoSample.key = true;
          VideoSample.frame = true;
          break;
        // SEI
        case 6: {
          push2 = true;
          parseSEIMessageFromNALu(unit.data, 1, pes.pts, textTrack.samples);
          break;
        }
        case 7: {
          var _track$pixelRatio, _track$pixelRatio2;
          push2 = true;
          spsfound = true;
          const sps = unit.data;
          const expGolombDecoder = new ExpGolomb(sps);
          const config = expGolombDecoder.readSPS();
          if (!track.sps || track.width !== config.width || track.height !== config.height || ((_track$pixelRatio = track.pixelRatio) == null ? void 0 : _track$pixelRatio[0]) !== config.pixelRatio[0] || ((_track$pixelRatio2 = track.pixelRatio) == null ? void 0 : _track$pixelRatio2[1]) !== config.pixelRatio[1]) {
            track.width = config.width;
            track.height = config.height;
            track.pixelRatio = config.pixelRatio;
            track.sps = [sps];
            track.duration = duration;
            const codecarray = sps.subarray(1, 4);
            let codecstring = "avc1.";
            for (let i2 = 0; i2 < 3; i2++) {
              let h = codecarray[i2].toString(16);
              if (h.length < 2) {
                h = "0" + h;
              }
              codecstring += h;
            }
            track.codec = codecstring;
          }
          break;
        }
        // PPS
        case 8:
          push2 = true;
          track.pps = [unit.data];
          break;
        // AUD
        case 9:
          push2 = true;
          track.audFound = true;
          if (VideoSample) {
            this.pushAccessUnit(VideoSample, track);
          }
          VideoSample = this.VideoSample = this.createVideoSample(false, pes.pts, pes.dts, "");
          break;
        // Filler Data
        case 12:
          push2 = true;
          break;
        default:
          push2 = false;
          if (VideoSample) {
            VideoSample.debug += "unknown NAL " + unit.type + " ";
          }
          break;
      }
      if (VideoSample && push2) {
        const units2 = VideoSample.units;
        units2.push(unit);
      }
    });
    if (last && VideoSample) {
      this.pushAccessUnit(VideoSample, track);
      this.VideoSample = null;
    }
  }
  parseAVCNALu(track, array) {
    const len = array.byteLength;
    let state = track.naluState || 0;
    const lastState = state;
    const units = [];
    let i2 = 0;
    let value;
    let overflow;
    let unitType;
    let lastUnitStart = -1;
    let lastUnitType = 0;
    if (state === -1) {
      lastUnitStart = 0;
      lastUnitType = array[0] & 31;
      state = 0;
      i2 = 1;
    }
    while (i2 < len) {
      value = array[i2++];
      if (!state) {
        state = value ? 0 : 1;
        continue;
      }
      if (state === 1) {
        state = value ? 0 : 2;
        continue;
      }
      if (!value) {
        state = 3;
      } else if (value === 1) {
        overflow = i2 - state - 1;
        if (lastUnitStart >= 0) {
          const unit = {
            data: array.subarray(lastUnitStart, overflow),
            type: lastUnitType
          };
          units.push(unit);
        } else {
          const lastUnit = this.getLastNalUnit(track.samples);
          if (lastUnit) {
            if (lastState && i2 <= 4 - lastState) {
              if (lastUnit.state) {
                lastUnit.data = lastUnit.data.subarray(0, lastUnit.data.byteLength - lastState);
              }
            }
            if (overflow > 0) {
              lastUnit.data = appendUint8Array(lastUnit.data, array.subarray(0, overflow));
              lastUnit.state = 0;
            }
          }
        }
        if (i2 < len) {
          unitType = array[i2] & 31;
          lastUnitStart = i2;
          lastUnitType = unitType;
          state = 0;
        } else {
          state = -1;
        }
      } else {
        state = 0;
      }
    }
    if (lastUnitStart >= 0 && state >= 0) {
      const unit = {
        data: array.subarray(lastUnitStart, len),
        type: lastUnitType,
        state
      };
      units.push(unit);
    }
    if (units.length === 0) {
      const lastUnit = this.getLastNalUnit(track.samples);
      if (lastUnit) {
        lastUnit.data = appendUint8Array(lastUnit.data, array);
      }
    }
    track.naluState = state;
    return units;
  }
}
class SampleAesDecrypter {
  constructor(observer, config, keyData) {
    this.keyData = void 0;
    this.decrypter = void 0;
    this.keyData = keyData;
    this.decrypter = new Decrypter(config, {
      removePKCS7Padding: false
    });
  }
  decryptBuffer(encryptedData) {
    return this.decrypter.decrypt(encryptedData, this.keyData.key.buffer, this.keyData.iv.buffer);
  }
  // AAC - encrypt all full 16 bytes blocks starting from offset 16
  decryptAacSample(samples, sampleIndex, callback) {
    const curUnit = samples[sampleIndex].unit;
    if (curUnit.length <= 16) {
      return;
    }
    const encryptedData = curUnit.subarray(16, curUnit.length - curUnit.length % 16);
    const encryptedBuffer = encryptedData.buffer.slice(encryptedData.byteOffset, encryptedData.byteOffset + encryptedData.length);
    this.decryptBuffer(encryptedBuffer).then((decryptedBuffer) => {
      const decryptedData = new Uint8Array(decryptedBuffer);
      curUnit.set(decryptedData, 16);
      if (!this.decrypter.isSync()) {
        this.decryptAacSamples(samples, sampleIndex + 1, callback);
      }
    });
  }
  decryptAacSamples(samples, sampleIndex, callback) {
    for (; ; sampleIndex++) {
      if (sampleIndex >= samples.length) {
        callback();
        return;
      }
      if (samples[sampleIndex].unit.length < 32) {
        continue;
      }
      this.decryptAacSample(samples, sampleIndex, callback);
      if (!this.decrypter.isSync()) {
        return;
      }
    }
  }
  // AVC - encrypt one 16 bytes block out of ten, starting from offset 32
  getAvcEncryptedData(decodedData) {
    const encryptedDataLen = Math.floor((decodedData.length - 48) / 160) * 16 + 16;
    const encryptedData = new Int8Array(encryptedDataLen);
    let outputPos = 0;
    for (let inputPos = 32; inputPos < decodedData.length - 16; inputPos += 160, outputPos += 16) {
      encryptedData.set(decodedData.subarray(inputPos, inputPos + 16), outputPos);
    }
    return encryptedData;
  }
  getAvcDecryptedUnit(decodedData, decryptedData) {
    const uint8DecryptedData = new Uint8Array(decryptedData);
    let inputPos = 0;
    for (let outputPos = 32; outputPos < decodedData.length - 16; outputPos += 160, inputPos += 16) {
      decodedData.set(uint8DecryptedData.subarray(inputPos, inputPos + 16), outputPos);
    }
    return decodedData;
  }
  decryptAvcSample(samples, sampleIndex, unitIndex, callback, curUnit) {
    const decodedData = discardEPB(curUnit.data);
    const encryptedData = this.getAvcEncryptedData(decodedData);
    this.decryptBuffer(encryptedData.buffer).then((decryptedBuffer) => {
      curUnit.data = this.getAvcDecryptedUnit(decodedData, decryptedBuffer);
      if (!this.decrypter.isSync()) {
        this.decryptAvcSamples(samples, sampleIndex, unitIndex + 1, callback);
      }
    });
  }
  decryptAvcSamples(samples, sampleIndex, unitIndex, callback) {
    if (samples instanceof Uint8Array) {
      throw new Error("Cannot decrypt samples of type Uint8Array");
    }
    for (; ; sampleIndex++, unitIndex = 0) {
      if (sampleIndex >= samples.length) {
        callback();
        return;
      }
      const curUnits = samples[sampleIndex].units;
      for (; ; unitIndex++) {
        if (unitIndex >= curUnits.length) {
          break;
        }
        const curUnit = curUnits[unitIndex];
        if (curUnit.data.length <= 48 || curUnit.type !== 1 && curUnit.type !== 5) {
          continue;
        }
        this.decryptAvcSample(samples, sampleIndex, unitIndex, callback, curUnit);
        if (!this.decrypter.isSync()) {
          return;
        }
      }
    }
  }
}
const PACKET_LENGTH = 188;
class TSDemuxer {
  constructor(observer, config, typeSupported) {
    this.observer = void 0;
    this.config = void 0;
    this.typeSupported = void 0;
    this.sampleAes = null;
    this.pmtParsed = false;
    this.audioCodec = void 0;
    this.videoCodec = void 0;
    this._duration = 0;
    this._pmtId = -1;
    this._videoTrack = void 0;
    this._audioTrack = void 0;
    this._id3Track = void 0;
    this._txtTrack = void 0;
    this.aacOverFlow = null;
    this.remainderData = null;
    this.videoParser = void 0;
    this.observer = observer;
    this.config = config;
    this.typeSupported = typeSupported;
    this.videoParser = new AvcVideoParser();
  }
  static probe(data) {
    const syncOffset = TSDemuxer.syncOffset(data);
    if (syncOffset > 0) {
      logger.warn(`MPEG2-TS detected but first sync word found @ offset ${syncOffset}`);
    }
    return syncOffset !== -1;
  }
  static syncOffset(data) {
    const length = data.length;
    let scanwindow = Math.min(PACKET_LENGTH * 5, length - PACKET_LENGTH) + 1;
    let i2 = 0;
    while (i2 < scanwindow) {
      let foundPat = false;
      let packetStart = -1;
      let tsPackets = 0;
      for (let j = i2; j < length; j += PACKET_LENGTH) {
        if (data[j] === 71 && (length - j === PACKET_LENGTH || data[j + PACKET_LENGTH] === 71)) {
          tsPackets++;
          if (packetStart === -1) {
            packetStart = j;
            if (packetStart !== 0) {
              scanwindow = Math.min(packetStart + PACKET_LENGTH * 99, data.length - PACKET_LENGTH) + 1;
            }
          }
          if (!foundPat) {
            foundPat = parsePID(data, j) === 0;
          }
          if (foundPat && tsPackets > 1 && (packetStart === 0 && tsPackets > 2 || j + PACKET_LENGTH > scanwindow)) {
            return packetStart;
          }
        } else if (tsPackets) {
          return -1;
        } else {
          break;
        }
      }
      i2++;
    }
    return -1;
  }
  /**
   * Creates a track model internal to demuxer used to drive remuxing input
   */
  static createTrack(type, duration) {
    return {
      container: type === "video" || type === "audio" ? "video/mp2t" : void 0,
      type,
      id: RemuxerTrackIdConfig[type],
      pid: -1,
      inputTimeScale: 9e4,
      sequenceNumber: 0,
      samples: [],
      dropped: 0,
      duration: type === "audio" ? duration : void 0
    };
  }
  /**
   * Initializes a new init segment on the demuxer/remuxer interface. Needed for discontinuities/track-switches (or at stream start)
   * Resets all internal track instances of the demuxer.
   */
  resetInitSegment(initSegment, audioCodec, videoCodec, trackDuration) {
    this.pmtParsed = false;
    this._pmtId = -1;
    this._videoTrack = TSDemuxer.createTrack("video");
    this._audioTrack = TSDemuxer.createTrack("audio", trackDuration);
    this._id3Track = TSDemuxer.createTrack("id3");
    this._txtTrack = TSDemuxer.createTrack("text");
    this._audioTrack.segmentCodec = "aac";
    this.aacOverFlow = null;
    this.remainderData = null;
    this.audioCodec = audioCodec;
    this.videoCodec = videoCodec;
    this._duration = trackDuration;
  }
  resetTimeStamp() {
  }
  resetContiguity() {
    const {
      _audioTrack,
      _videoTrack,
      _id3Track
    } = this;
    if (_audioTrack) {
      _audioTrack.pesData = null;
    }
    if (_videoTrack) {
      _videoTrack.pesData = null;
    }
    if (_id3Track) {
      _id3Track.pesData = null;
    }
    this.aacOverFlow = null;
    this.remainderData = null;
  }
  demux(data, timeOffset, isSampleAes = false, flush = false) {
    if (!isSampleAes) {
      this.sampleAes = null;
    }
    let pes;
    const videoTrack = this._videoTrack;
    const audioTrack = this._audioTrack;
    const id3Track = this._id3Track;
    const textTrack = this._txtTrack;
    let videoPid = videoTrack.pid;
    let videoData = videoTrack.pesData;
    let audioPid = audioTrack.pid;
    let id3Pid = id3Track.pid;
    let audioData = audioTrack.pesData;
    let id3Data = id3Track.pesData;
    let unknownPID = null;
    let pmtParsed = this.pmtParsed;
    let pmtId = this._pmtId;
    let len = data.length;
    if (this.remainderData) {
      data = appendUint8Array(this.remainderData, data);
      len = data.length;
      this.remainderData = null;
    }
    if (len < PACKET_LENGTH && !flush) {
      this.remainderData = data;
      return {
        audioTrack,
        videoTrack,
        id3Track,
        textTrack
      };
    }
    const syncOffset = Math.max(0, TSDemuxer.syncOffset(data));
    len -= (len - syncOffset) % PACKET_LENGTH;
    if (len < data.byteLength && !flush) {
      this.remainderData = new Uint8Array(data.buffer, len, data.buffer.byteLength - len);
    }
    let tsPacketErrors = 0;
    for (let start = syncOffset; start < len; start += PACKET_LENGTH) {
      if (data[start] === 71) {
        const stt = !!(data[start + 1] & 64);
        const pid = parsePID(data, start);
        const atf = (data[start + 3] & 48) >> 4;
        let offset;
        if (atf > 1) {
          offset = start + 5 + data[start + 4];
          if (offset === start + PACKET_LENGTH) {
            continue;
          }
        } else {
          offset = start + 4;
        }
        switch (pid) {
          case videoPid:
            if (stt) {
              if (videoData && (pes = parsePES(videoData))) {
                this.videoParser.parseAVCPES(videoTrack, textTrack, pes, false, this._duration);
              }
              videoData = {
                data: [],
                size: 0
              };
            }
            if (videoData) {
              videoData.data.push(data.subarray(offset, start + PACKET_LENGTH));
              videoData.size += start + PACKET_LENGTH - offset;
            }
            break;
          case audioPid:
            if (stt) {
              if (audioData && (pes = parsePES(audioData))) {
                switch (audioTrack.segmentCodec) {
                  case "aac":
                    this.parseAACPES(audioTrack, pes);
                    break;
                  case "mp3":
                    this.parseMPEGPES(audioTrack, pes);
                    break;
                  case "ac3":
                    {
                      this.parseAC3PES(audioTrack, pes);
                    }
                    break;
                }
              }
              audioData = {
                data: [],
                size: 0
              };
            }
            if (audioData) {
              audioData.data.push(data.subarray(offset, start + PACKET_LENGTH));
              audioData.size += start + PACKET_LENGTH - offset;
            }
            break;
          case id3Pid:
            if (stt) {
              if (id3Data && (pes = parsePES(id3Data))) {
                this.parseID3PES(id3Track, pes);
              }
              id3Data = {
                data: [],
                size: 0
              };
            }
            if (id3Data) {
              id3Data.data.push(data.subarray(offset, start + PACKET_LENGTH));
              id3Data.size += start + PACKET_LENGTH - offset;
            }
            break;
          case 0:
            if (stt) {
              offset += data[offset] + 1;
            }
            pmtId = this._pmtId = parsePAT(data, offset);
            break;
          case pmtId: {
            if (stt) {
              offset += data[offset] + 1;
            }
            const parsedPIDs = parsePMT(data, offset, this.typeSupported, isSampleAes, this.observer);
            videoPid = parsedPIDs.videoPid;
            if (videoPid > 0) {
              videoTrack.pid = videoPid;
              videoTrack.segmentCodec = parsedPIDs.segmentVideoCodec;
            }
            audioPid = parsedPIDs.audioPid;
            if (audioPid > 0) {
              audioTrack.pid = audioPid;
              audioTrack.segmentCodec = parsedPIDs.segmentAudioCodec;
            }
            id3Pid = parsedPIDs.id3Pid;
            if (id3Pid > 0) {
              id3Track.pid = id3Pid;
            }
            if (unknownPID !== null && !pmtParsed) {
              logger.warn(`MPEG-TS PMT found at ${start} after unknown PID '${unknownPID}'. Backtracking to sync byte @${syncOffset} to parse all TS packets.`);
              unknownPID = null;
              start = syncOffset - 188;
            }
            pmtParsed = this.pmtParsed = true;
            break;
          }
          case 17:
          case 8191:
            break;
          default:
            unknownPID = pid;
            break;
        }
      } else {
        tsPacketErrors++;
      }
    }
    if (tsPacketErrors > 0) {
      emitParsingError(this.observer, new Error(`Found ${tsPacketErrors} TS packet/s that do not start with 0x47`));
    }
    videoTrack.pesData = videoData;
    audioTrack.pesData = audioData;
    id3Track.pesData = id3Data;
    const demuxResult = {
      audioTrack,
      videoTrack,
      id3Track,
      textTrack
    };
    if (flush) {
      this.extractRemainingSamples(demuxResult);
    }
    return demuxResult;
  }
  flush() {
    const {
      remainderData
    } = this;
    this.remainderData = null;
    let result;
    if (remainderData) {
      result = this.demux(remainderData, -1, false, true);
    } else {
      result = {
        videoTrack: this._videoTrack,
        audioTrack: this._audioTrack,
        id3Track: this._id3Track,
        textTrack: this._txtTrack
      };
    }
    this.extractRemainingSamples(result);
    if (this.sampleAes) {
      return this.decrypt(result, this.sampleAes);
    }
    return result;
  }
  extractRemainingSamples(demuxResult) {
    const {
      audioTrack,
      videoTrack,
      id3Track,
      textTrack
    } = demuxResult;
    const videoData = videoTrack.pesData;
    const audioData = audioTrack.pesData;
    const id3Data = id3Track.pesData;
    let pes;
    if (videoData && (pes = parsePES(videoData))) {
      this.videoParser.parseAVCPES(videoTrack, textTrack, pes, true, this._duration);
      videoTrack.pesData = null;
    } else {
      videoTrack.pesData = videoData;
    }
    if (audioData && (pes = parsePES(audioData))) {
      switch (audioTrack.segmentCodec) {
        case "aac":
          this.parseAACPES(audioTrack, pes);
          break;
        case "mp3":
          this.parseMPEGPES(audioTrack, pes);
          break;
        case "ac3":
          {
            this.parseAC3PES(audioTrack, pes);
          }
          break;
      }
      audioTrack.pesData = null;
    } else {
      if (audioData != null && audioData.size) {
        logger.log("last AAC PES packet truncated,might overlap between fragments");
      }
      audioTrack.pesData = audioData;
    }
    if (id3Data && (pes = parsePES(id3Data))) {
      this.parseID3PES(id3Track, pes);
      id3Track.pesData = null;
    } else {
      id3Track.pesData = id3Data;
    }
  }
  demuxSampleAes(data, keyData, timeOffset) {
    const demuxResult = this.demux(data, timeOffset, true, !this.config.progressive);
    const sampleAes = this.sampleAes = new SampleAesDecrypter(this.observer, this.config, keyData);
    return this.decrypt(demuxResult, sampleAes);
  }
  decrypt(demuxResult, sampleAes) {
    return new Promise((resolve) => {
      const {
        audioTrack,
        videoTrack
      } = demuxResult;
      if (audioTrack.samples && audioTrack.segmentCodec === "aac") {
        sampleAes.decryptAacSamples(audioTrack.samples, 0, () => {
          if (videoTrack.samples) {
            sampleAes.decryptAvcSamples(videoTrack.samples, 0, 0, () => {
              resolve(demuxResult);
            });
          } else {
            resolve(demuxResult);
          }
        });
      } else if (videoTrack.samples) {
        sampleAes.decryptAvcSamples(videoTrack.samples, 0, 0, () => {
          resolve(demuxResult);
        });
      }
    });
  }
  destroy() {
    this._duration = 0;
  }
  parseAACPES(track, pes) {
    let startOffset = 0;
    const aacOverFlow = this.aacOverFlow;
    let data = pes.data;
    if (aacOverFlow) {
      this.aacOverFlow = null;
      const frameMissingBytes = aacOverFlow.missing;
      const sampleLength = aacOverFlow.sample.unit.byteLength;
      if (frameMissingBytes === -1) {
        data = appendUint8Array(aacOverFlow.sample.unit, data);
      } else {
        const frameOverflowBytes = sampleLength - frameMissingBytes;
        aacOverFlow.sample.unit.set(data.subarray(0, frameMissingBytes), frameOverflowBytes);
        track.samples.push(aacOverFlow.sample);
        startOffset = aacOverFlow.missing;
      }
    }
    let offset;
    let len;
    for (offset = startOffset, len = data.length; offset < len - 1; offset++) {
      if (isHeader$1(data, offset)) {
        break;
      }
    }
    if (offset !== startOffset) {
      let reason;
      const recoverable = offset < len - 1;
      if (recoverable) {
        reason = `AAC PES did not start with ADTS header,offset:${offset}`;
      } else {
        reason = "No ADTS header found in AAC PES";
      }
      emitParsingError(this.observer, new Error(reason), recoverable);
      if (!recoverable) {
        return;
      }
    }
    initTrackConfig(track, this.observer, data, offset, this.audioCodec);
    let pts;
    if (pes.pts !== void 0) {
      pts = pes.pts;
    } else if (aacOverFlow) {
      const frameDuration = getFrameDuration(track.samplerate);
      pts = aacOverFlow.sample.pts + frameDuration;
    } else {
      logger.warn("[tsdemuxer]: AAC PES unknown PTS");
      return;
    }
    let frameIndex = 0;
    let frame;
    while (offset < len) {
      frame = appendFrame$2(track, data, offset, pts, frameIndex);
      offset += frame.length;
      if (!frame.missing) {
        frameIndex++;
        for (; offset < len - 1; offset++) {
          if (isHeader$1(data, offset)) {
            break;
          }
        }
      } else {
        this.aacOverFlow = frame;
        break;
      }
    }
  }
  parseMPEGPES(track, pes) {
    const data = pes.data;
    const length = data.length;
    let frameIndex = 0;
    let offset = 0;
    const pts = pes.pts;
    if (pts === void 0) {
      logger.warn("[tsdemuxer]: MPEG PES unknown PTS");
      return;
    }
    while (offset < length) {
      if (isHeader(data, offset)) {
        const frame = appendFrame$1(track, data, offset, pts, frameIndex);
        if (frame) {
          offset += frame.length;
          frameIndex++;
        } else {
          break;
        }
      } else {
        offset++;
      }
    }
  }
  parseAC3PES(track, pes) {
    {
      const data = pes.data;
      const pts = pes.pts;
      if (pts === void 0) {
        logger.warn("[tsdemuxer]: AC3 PES unknown PTS");
        return;
      }
      const length = data.length;
      let frameIndex = 0;
      let offset = 0;
      let parsed;
      while (offset < length && (parsed = appendFrame(track, data, offset, pts, frameIndex++)) > 0) {
        offset += parsed;
      }
    }
  }
  parseID3PES(id3Track, pes) {
    if (pes.pts === void 0) {
      logger.warn("[tsdemuxer]: ID3 PES unknown PTS");
      return;
    }
    const id3Sample = _extends({}, pes, {
      type: this._videoTrack ? MetadataSchema.emsg : MetadataSchema.audioId3,
      duration: Number.POSITIVE_INFINITY
    });
    id3Track.samples.push(id3Sample);
  }
}
function parsePID(data, offset) {
  return ((data[offset + 1] & 31) << 8) + data[offset + 2];
}
function parsePAT(data, offset) {
  return (data[offset + 10] & 31) << 8 | data[offset + 11];
}
function parsePMT(data, offset, typeSupported, isSampleAes, observer) {
  const result = {
    audioPid: -1,
    videoPid: -1,
    id3Pid: -1,
    segmentVideoCodec: "avc",
    segmentAudioCodec: "aac"
  };
  const sectionLength = (data[offset + 1] & 15) << 8 | data[offset + 2];
  const tableEnd = offset + 3 + sectionLength - 4;
  const programInfoLength = (data[offset + 10] & 15) << 8 | data[offset + 11];
  offset += 12 + programInfoLength;
  while (offset < tableEnd) {
    const pid = parsePID(data, offset);
    const esInfoLength = (data[offset + 3] & 15) << 8 | data[offset + 4];
    switch (data[offset]) {
      case 207:
        if (!isSampleAes) {
          logEncryptedSamplesFoundInUnencryptedStream("ADTS AAC");
          break;
        }
      /* falls through */
      case 15:
        if (result.audioPid === -1) {
          result.audioPid = pid;
        }
        break;
      // Packetized metadata (ID3)
      case 21:
        if (result.id3Pid === -1) {
          result.id3Pid = pid;
        }
        break;
      case 219:
        if (!isSampleAes) {
          logEncryptedSamplesFoundInUnencryptedStream("H.264");
          break;
        }
      /* falls through */
      case 27:
        if (result.videoPid === -1) {
          result.videoPid = pid;
          result.segmentVideoCodec = "avc";
        }
        break;
      // ISO/IEC 11172-3 (MPEG-1 audio)
      // or ISO/IEC 13818-3 (MPEG-2 halved sample rate audio)
      case 3:
      case 4:
        if (!typeSupported.mpeg && !typeSupported.mp3) {
          logger.log("MPEG audio found, not supported in this browser");
        } else if (result.audioPid === -1) {
          result.audioPid = pid;
          result.segmentAudioCodec = "mp3";
        }
        break;
      case 193:
        if (!isSampleAes) {
          logEncryptedSamplesFoundInUnencryptedStream("AC-3");
          break;
        }
      /* falls through */
      case 129:
        {
          if (!typeSupported.ac3) {
            logger.log("AC-3 audio found, not supported in this browser");
          } else if (result.audioPid === -1) {
            result.audioPid = pid;
            result.segmentAudioCodec = "ac3";
          }
        }
        break;
      case 6:
        if (result.audioPid === -1 && esInfoLength > 0) {
          let parsePos = offset + 5;
          let remaining = esInfoLength;
          while (remaining > 2) {
            const descriptorId = data[parsePos];
            switch (descriptorId) {
              case 106:
                {
                  if (typeSupported.ac3 !== true) {
                    logger.log("AC-3 audio found, not supported in this browser for now");
                  } else {
                    result.audioPid = pid;
                    result.segmentAudioCodec = "ac3";
                  }
                }
                break;
            }
            const descriptorLen = data[parsePos + 1] + 2;
            parsePos += descriptorLen;
            remaining -= descriptorLen;
          }
        }
        break;
      case 194:
      // SAMPLE-AES EC3
      /* falls through */
      case 135:
        emitParsingError(observer, new Error("Unsupported EC-3 in M2TS found"));
        return result;
      case 36:
        emitParsingError(observer, new Error("Unsupported HEVC in M2TS found"));
        return result;
    }
    offset += esInfoLength + 5;
  }
  return result;
}
function emitParsingError(observer, error, levelRetry) {
  logger.warn(`parsing error: ${error.message}`);
  observer.emit(Events.ERROR, Events.ERROR, {
    type: ErrorTypes.MEDIA_ERROR,
    details: ErrorDetails.FRAG_PARSING_ERROR,
    fatal: false,
    levelRetry,
    error,
    reason: error.message
  });
}
function logEncryptedSamplesFoundInUnencryptedStream(type) {
  logger.log(`${type} with AES-128-CBC encryption found in unencrypted stream`);
}
function parsePES(stream) {
  let i2 = 0;
  let frag;
  let pesLen;
  let pesHdrLen;
  let pesPts;
  let pesDts;
  const data = stream.data;
  if (!stream || stream.size === 0) {
    return null;
  }
  while (data[0].length < 19 && data.length > 1) {
    data[0] = appendUint8Array(data[0], data[1]);
    data.splice(1, 1);
  }
  frag = data[0];
  const pesPrefix = (frag[0] << 16) + (frag[1] << 8) + frag[2];
  if (pesPrefix === 1) {
    pesLen = (frag[4] << 8) + frag[5];
    if (pesLen && pesLen > stream.size - 6) {
      return null;
    }
    const pesFlags = frag[7];
    if (pesFlags & 192) {
      pesPts = (frag[9] & 14) * 536870912 + // 1 << 29
      (frag[10] & 255) * 4194304 + // 1 << 22
      (frag[11] & 254) * 16384 + // 1 << 14
      (frag[12] & 255) * 128 + // 1 << 7
      (frag[13] & 254) / 2;
      if (pesFlags & 64) {
        pesDts = (frag[14] & 14) * 536870912 + // 1 << 29
        (frag[15] & 255) * 4194304 + // 1 << 22
        (frag[16] & 254) * 16384 + // 1 << 14
        (frag[17] & 255) * 128 + // 1 << 7
        (frag[18] & 254) / 2;
        if (pesPts - pesDts > 60 * 9e4) {
          logger.warn(`${Math.round((pesPts - pesDts) / 9e4)}s delta between PTS and DTS, align them`);
          pesPts = pesDts;
        }
      } else {
        pesDts = pesPts;
      }
    }
    pesHdrLen = frag[8];
    let payloadStartOffset = pesHdrLen + 9;
    if (stream.size <= payloadStartOffset) {
      return null;
    }
    stream.size -= payloadStartOffset;
    const pesData = new Uint8Array(stream.size);
    for (let j = 0, dataLen = data.length; j < dataLen; j++) {
      frag = data[j];
      let len = frag.byteLength;
      if (payloadStartOffset) {
        if (payloadStartOffset > len) {
          payloadStartOffset -= len;
          continue;
        } else {
          frag = frag.subarray(payloadStartOffset);
          len -= payloadStartOffset;
          payloadStartOffset = 0;
        }
      }
      pesData.set(frag, i2);
      i2 += len;
    }
    if (pesLen) {
      pesLen -= pesHdrLen + 3;
    }
    return {
      data: pesData,
      pts: pesPts,
      dts: pesDts,
      len: pesLen
    };
  }
  return null;
}
class MP3Demuxer extends BaseAudioDemuxer {
  resetInitSegment(initSegment, audioCodec, videoCodec, trackDuration) {
    super.resetInitSegment(initSegment, audioCodec, videoCodec, trackDuration);
    this._audioTrack = {
      container: "audio/mpeg",
      type: "audio",
      id: 2,
      pid: -1,
      sequenceNumber: 0,
      segmentCodec: "mp3",
      samples: [],
      manifestCodec: audioCodec,
      duration: trackDuration,
      inputTimeScale: 9e4,
      dropped: 0
    };
  }
  static probe(data) {
    if (!data) {
      return false;
    }
    const id3Data = getID3Data(data, 0);
    let offset = (id3Data == null ? void 0 : id3Data.length) || 0;
    if (id3Data && data[offset] === 11 && data[offset + 1] === 119 && getTimeStamp(id3Data) !== void 0 && // check the bsid to confirm ac-3 or ec-3 (not mp3)
    getAudioBSID(data, offset) <= 16) {
      return false;
    }
    for (let length = data.length; offset < length; offset++) {
      if (probe(data, offset)) {
        logger.log("MPEG Audio sync word found !");
        return true;
      }
    }
    return false;
  }
  canParse(data, offset) {
    return canParse(data, offset);
  }
  appendFrame(track, data, offset) {
    if (this.basePTS === null) {
      return;
    }
    return appendFrame$1(track, data, offset, this.basePTS, this.frameIndex);
  }
}
class AAC {
  static getSilentFrame(codec, channelCount) {
    switch (codec) {
      case "mp4a.40.2":
        if (channelCount === 1) {
          return new Uint8Array([0, 200, 0, 128, 35, 128]);
        } else if (channelCount === 2) {
          return new Uint8Array([33, 0, 73, 144, 2, 25, 0, 35, 128]);
        } else if (channelCount === 3) {
          return new Uint8Array([0, 200, 0, 128, 32, 132, 1, 38, 64, 8, 100, 0, 142]);
        } else if (channelCount === 4) {
          return new Uint8Array([0, 200, 0, 128, 32, 132, 1, 38, 64, 8, 100, 0, 128, 44, 128, 8, 2, 56]);
        } else if (channelCount === 5) {
          return new Uint8Array([0, 200, 0, 128, 32, 132, 1, 38, 64, 8, 100, 0, 130, 48, 4, 153, 0, 33, 144, 2, 56]);
        } else if (channelCount === 6) {
          return new Uint8Array([0, 200, 0, 128, 32, 132, 1, 38, 64, 8, 100, 0, 130, 48, 4, 153, 0, 33, 144, 2, 0, 178, 0, 32, 8, 224]);
        }
        break;
      // handle HE-AAC below (mp4a.40.5 / mp4a.40.29)
      default:
        if (channelCount === 1) {
          return new Uint8Array([1, 64, 34, 128, 163, 78, 230, 128, 186, 8, 0, 0, 0, 28, 6, 241, 193, 10, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 94]);
        } else if (channelCount === 2) {
          return new Uint8Array([1, 64, 34, 128, 163, 94, 230, 128, 186, 8, 0, 0, 0, 0, 149, 0, 6, 241, 161, 10, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 94]);
        } else if (channelCount === 3) {
          return new Uint8Array([1, 64, 34, 128, 163, 94, 230, 128, 186, 8, 0, 0, 0, 0, 149, 0, 6, 241, 161, 10, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 94]);
        }
        break;
    }
    return void 0;
  }
}
const UINT32_MAX = Math.pow(2, 32) - 1;
class MP4 {
  static init() {
    MP4.types = {
      avc1: [],
      // codingname
      avcC: [],
      btrt: [],
      dinf: [],
      dref: [],
      esds: [],
      ftyp: [],
      hdlr: [],
      mdat: [],
      mdhd: [],
      mdia: [],
      mfhd: [],
      minf: [],
      moof: [],
      moov: [],
      mp4a: [],
      ".mp3": [],
      dac3: [],
      "ac-3": [],
      mvex: [],
      mvhd: [],
      pasp: [],
      sdtp: [],
      stbl: [],
      stco: [],
      stsc: [],
      stsd: [],
      stsz: [],
      stts: [],
      tfdt: [],
      tfhd: [],
      traf: [],
      trak: [],
      trun: [],
      trex: [],
      tkhd: [],
      vmhd: [],
      smhd: []
    };
    let i2;
    for (i2 in MP4.types) {
      if (MP4.types.hasOwnProperty(i2)) {
        MP4.types[i2] = [i2.charCodeAt(0), i2.charCodeAt(1), i2.charCodeAt(2), i2.charCodeAt(3)];
      }
    }
    const videoHdlr = new Uint8Array([
      0,
      // version 0
      0,
      0,
      0,
      // flags
      0,
      0,
      0,
      0,
      // pre_defined
      118,
      105,
      100,
      101,
      // handler_type: 'vide'
      0,
      0,
      0,
      0,
      // reserved
      0,
      0,
      0,
      0,
      // reserved
      0,
      0,
      0,
      0,
      // reserved
      86,
      105,
      100,
      101,
      111,
      72,
      97,
      110,
      100,
      108,
      101,
      114,
      0
      // name: 'VideoHandler'
    ]);
    const audioHdlr = new Uint8Array([
      0,
      // version 0
      0,
      0,
      0,
      // flags
      0,
      0,
      0,
      0,
      // pre_defined
      115,
      111,
      117,
      110,
      // handler_type: 'soun'
      0,
      0,
      0,
      0,
      // reserved
      0,
      0,
      0,
      0,
      // reserved
      0,
      0,
      0,
      0,
      // reserved
      83,
      111,
      117,
      110,
      100,
      72,
      97,
      110,
      100,
      108,
      101,
      114,
      0
      // name: 'SoundHandler'
    ]);
    MP4.HDLR_TYPES = {
      video: videoHdlr,
      audio: audioHdlr
    };
    const dref = new Uint8Array([
      0,
      // version 0
      0,
      0,
      0,
      // flags
      0,
      0,
      0,
      1,
      // entry_count
      0,
      0,
      0,
      12,
      // entry_size
      117,
      114,
      108,
      32,
      // 'url' type
      0,
      // version 0
      0,
      0,
      1
      // entry_flags
    ]);
    const stco = new Uint8Array([
      0,
      // version
      0,
      0,
      0,
      // flags
      0,
      0,
      0,
      0
      // entry_count
    ]);
    MP4.STTS = MP4.STSC = MP4.STCO = stco;
    MP4.STSZ = new Uint8Array([
      0,
      // version
      0,
      0,
      0,
      // flags
      0,
      0,
      0,
      0,
      // sample_size
      0,
      0,
      0,
      0
      // sample_count
    ]);
    MP4.VMHD = new Uint8Array([
      0,
      // version
      0,
      0,
      1,
      // flags
      0,
      0,
      // graphicsmode
      0,
      0,
      0,
      0,
      0,
      0
      // opcolor
    ]);
    MP4.SMHD = new Uint8Array([
      0,
      // version
      0,
      0,
      0,
      // flags
      0,
      0,
      // balance
      0,
      0
      // reserved
    ]);
    MP4.STSD = new Uint8Array([
      0,
      // version 0
      0,
      0,
      0,
      // flags
      0,
      0,
      0,
      1
    ]);
    const majorBrand = new Uint8Array([105, 115, 111, 109]);
    const avc1Brand = new Uint8Array([97, 118, 99, 49]);
    const minorVersion = new Uint8Array([0, 0, 0, 1]);
    MP4.FTYP = MP4.box(MP4.types.ftyp, majorBrand, minorVersion, majorBrand, avc1Brand);
    MP4.DINF = MP4.box(MP4.types.dinf, MP4.box(MP4.types.dref, dref));
  }
  static box(type, ...payload) {
    let size = 8;
    let i2 = payload.length;
    const len = i2;
    while (i2--) {
      size += payload[i2].byteLength;
    }
    const result = new Uint8Array(size);
    result[0] = size >> 24 & 255;
    result[1] = size >> 16 & 255;
    result[2] = size >> 8 & 255;
    result[3] = size & 255;
    result.set(type, 4);
    for (i2 = 0, size = 8; i2 < len; i2++) {
      result.set(payload[i2], size);
      size += payload[i2].byteLength;
    }
    return result;
  }
  static hdlr(type) {
    return MP4.box(MP4.types.hdlr, MP4.HDLR_TYPES[type]);
  }
  static mdat(data) {
    return MP4.box(MP4.types.mdat, data);
  }
  static mdhd(timescale, duration) {
    duration *= timescale;
    const upperWordDuration = Math.floor(duration / (UINT32_MAX + 1));
    const lowerWordDuration = Math.floor(duration % (UINT32_MAX + 1));
    return MP4.box(MP4.types.mdhd, new Uint8Array([
      1,
      // version 1
      0,
      0,
      0,
      // flags
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      2,
      // creation_time
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      3,
      // modification_time
      timescale >> 24 & 255,
      timescale >> 16 & 255,
      timescale >> 8 & 255,
      timescale & 255,
      // timescale
      upperWordDuration >> 24,
      upperWordDuration >> 16 & 255,
      upperWordDuration >> 8 & 255,
      upperWordDuration & 255,
      lowerWordDuration >> 24,
      lowerWordDuration >> 16 & 255,
      lowerWordDuration >> 8 & 255,
      lowerWordDuration & 255,
      85,
      196,
      // 'und' language (undetermined)
      0,
      0
    ]));
  }
  static mdia(track) {
    return MP4.box(MP4.types.mdia, MP4.mdhd(track.timescale, track.duration), MP4.hdlr(track.type), MP4.minf(track));
  }
  static mfhd(sequenceNumber) {
    return MP4.box(MP4.types.mfhd, new Uint8Array([
      0,
      0,
      0,
      0,
      // flags
      sequenceNumber >> 24,
      sequenceNumber >> 16 & 255,
      sequenceNumber >> 8 & 255,
      sequenceNumber & 255
      // sequence_number
    ]));
  }
  static minf(track) {
    if (track.type === "audio") {
      return MP4.box(MP4.types.minf, MP4.box(MP4.types.smhd, MP4.SMHD), MP4.DINF, MP4.stbl(track));
    } else {
      return MP4.box(MP4.types.minf, MP4.box(MP4.types.vmhd, MP4.VMHD), MP4.DINF, MP4.stbl(track));
    }
  }
  static moof(sn, baseMediaDecodeTime, track) {
    return MP4.box(MP4.types.moof, MP4.mfhd(sn), MP4.traf(track, baseMediaDecodeTime));
  }
  static moov(tracks) {
    let i2 = tracks.length;
    const boxes = [];
    while (i2--) {
      boxes[i2] = MP4.trak(tracks[i2]);
    }
    return MP4.box.apply(null, [MP4.types.moov, MP4.mvhd(tracks[0].timescale, tracks[0].duration)].concat(boxes).concat(MP4.mvex(tracks)));
  }
  static mvex(tracks) {
    let i2 = tracks.length;
    const boxes = [];
    while (i2--) {
      boxes[i2] = MP4.trex(tracks[i2]);
    }
    return MP4.box.apply(null, [MP4.types.mvex, ...boxes]);
  }
  static mvhd(timescale, duration) {
    duration *= timescale;
    const upperWordDuration = Math.floor(duration / (UINT32_MAX + 1));
    const lowerWordDuration = Math.floor(duration % (UINT32_MAX + 1));
    const bytes = new Uint8Array([
      1,
      // version 1
      0,
      0,
      0,
      // flags
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      2,
      // creation_time
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      3,
      // modification_time
      timescale >> 24 & 255,
      timescale >> 16 & 255,
      timescale >> 8 & 255,
      timescale & 255,
      // timescale
      upperWordDuration >> 24,
      upperWordDuration >> 16 & 255,
      upperWordDuration >> 8 & 255,
      upperWordDuration & 255,
      lowerWordDuration >> 24,
      lowerWordDuration >> 16 & 255,
      lowerWordDuration >> 8 & 255,
      lowerWordDuration & 255,
      0,
      1,
      0,
      0,
      // 1.0 rate
      1,
      0,
      // 1.0 volume
      0,
      0,
      // reserved
      0,
      0,
      0,
      0,
      // reserved
      0,
      0,
      0,
      0,
      // reserved
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      64,
      0,
      0,
      0,
      // transformation: unity matrix
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      // pre_defined
      255,
      255,
      255,
      255
      // next_track_ID
    ]);
    return MP4.box(MP4.types.mvhd, bytes);
  }
  static sdtp(track) {
    const samples = track.samples || [];
    const bytes = new Uint8Array(4 + samples.length);
    let i2;
    let flags;
    for (i2 = 0; i2 < samples.length; i2++) {
      flags = samples[i2].flags;
      bytes[i2 + 4] = flags.dependsOn << 4 | flags.isDependedOn << 2 | flags.hasRedundancy;
    }
    return MP4.box(MP4.types.sdtp, bytes);
  }
  static stbl(track) {
    return MP4.box(MP4.types.stbl, MP4.stsd(track), MP4.box(MP4.types.stts, MP4.STTS), MP4.box(MP4.types.stsc, MP4.STSC), MP4.box(MP4.types.stsz, MP4.STSZ), MP4.box(MP4.types.stco, MP4.STCO));
  }
  static avc1(track) {
    let sps = [];
    let pps = [];
    let i2;
    let data;
    let len;
    for (i2 = 0; i2 < track.sps.length; i2++) {
      data = track.sps[i2];
      len = data.byteLength;
      sps.push(len >>> 8 & 255);
      sps.push(len & 255);
      sps = sps.concat(Array.prototype.slice.call(data));
    }
    for (i2 = 0; i2 < track.pps.length; i2++) {
      data = track.pps[i2];
      len = data.byteLength;
      pps.push(len >>> 8 & 255);
      pps.push(len & 255);
      pps = pps.concat(Array.prototype.slice.call(data));
    }
    const avcc = MP4.box(MP4.types.avcC, new Uint8Array([
      1,
      // version
      sps[3],
      // profile
      sps[4],
      // profile compat
      sps[5],
      // level
      252 | 3,
      // lengthSizeMinusOne, hard-coded to 4 bytes
      224 | track.sps.length
      // 3bit reserved (111) + numOfSequenceParameterSets
    ].concat(sps).concat([
      track.pps.length
      // numOfPictureParameterSets
    ]).concat(pps)));
    const width = track.width;
    const height = track.height;
    const hSpacing = track.pixelRatio[0];
    const vSpacing = track.pixelRatio[1];
    return MP4.box(
      MP4.types.avc1,
      new Uint8Array([
        0,
        0,
        0,
        // reserved
        0,
        0,
        0,
        // reserved
        0,
        1,
        // data_reference_index
        0,
        0,
        // pre_defined
        0,
        0,
        // reserved
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        // pre_defined
        width >> 8 & 255,
        width & 255,
        // width
        height >> 8 & 255,
        height & 255,
        // height
        0,
        72,
        0,
        0,
        // horizresolution
        0,
        72,
        0,
        0,
        // vertresolution
        0,
        0,
        0,
        0,
        // reserved
        0,
        1,
        // frame_count
        18,
        100,
        97,
        105,
        108,
        // dailymotion/hls.js
        121,
        109,
        111,
        116,
        105,
        111,
        110,
        47,
        104,
        108,
        115,
        46,
        106,
        115,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        // compressorname
        0,
        24,
        // depth = 24
        17,
        17
      ]),
      // pre_defined = -1
      avcc,
      MP4.box(MP4.types.btrt, new Uint8Array([
        0,
        28,
        156,
        128,
        // bufferSizeDB
        0,
        45,
        198,
        192,
        // maxBitrate
        0,
        45,
        198,
        192
      ])),
      // avgBitrate
      MP4.box(MP4.types.pasp, new Uint8Array([
        hSpacing >> 24,
        // hSpacing
        hSpacing >> 16 & 255,
        hSpacing >> 8 & 255,
        hSpacing & 255,
        vSpacing >> 24,
        // vSpacing
        vSpacing >> 16 & 255,
        vSpacing >> 8 & 255,
        vSpacing & 255
      ]))
    );
  }
  static esds(track) {
    const configlen = track.config.length;
    return new Uint8Array([
      0,
      // version 0
      0,
      0,
      0,
      // flags
      3,
      // descriptor_type
      23 + configlen,
      // length
      0,
      1,
      // es_id
      0,
      // stream_priority
      4,
      // descriptor_type
      15 + configlen,
      // length
      64,
      // codec : mpeg4_audio
      21,
      // stream_type
      0,
      0,
      0,
      // buffer_size
      0,
      0,
      0,
      0,
      // maxBitrate
      0,
      0,
      0,
      0,
      // avgBitrate
      5
      // descriptor_type
    ].concat([configlen]).concat(track.config).concat([6, 1, 2]));
  }
  static audioStsd(track) {
    const samplerate = track.samplerate;
    return new Uint8Array([
      0,
      0,
      0,
      // reserved
      0,
      0,
      0,
      // reserved
      0,
      1,
      // data_reference_index
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      // reserved
      0,
      track.channelCount,
      // channelcount
      0,
      16,
      // sampleSize:16bits
      0,
      0,
      0,
      0,
      // reserved2
      samplerate >> 8 & 255,
      samplerate & 255,
      //
      0,
      0
    ]);
  }
  static mp4a(track) {
    return MP4.box(MP4.types.mp4a, MP4.audioStsd(track), MP4.box(MP4.types.esds, MP4.esds(track)));
  }
  static mp3(track) {
    return MP4.box(MP4.types[".mp3"], MP4.audioStsd(track));
  }
  static ac3(track) {
    return MP4.box(MP4.types["ac-3"], MP4.audioStsd(track), MP4.box(MP4.types.dac3, track.config));
  }
  static stsd(track) {
    if (track.type === "audio") {
      if (track.segmentCodec === "mp3" && track.codec === "mp3") {
        return MP4.box(MP4.types.stsd, MP4.STSD, MP4.mp3(track));
      }
      if (track.segmentCodec === "ac3") {
        return MP4.box(MP4.types.stsd, MP4.STSD, MP4.ac3(track));
      }
      return MP4.box(MP4.types.stsd, MP4.STSD, MP4.mp4a(track));
    } else {
      return MP4.box(MP4.types.stsd, MP4.STSD, MP4.avc1(track));
    }
  }
  static tkhd(track) {
    const id = track.id;
    const duration = track.duration * track.timescale;
    const width = track.width;
    const height = track.height;
    const upperWordDuration = Math.floor(duration / (UINT32_MAX + 1));
    const lowerWordDuration = Math.floor(duration % (UINT32_MAX + 1));
    return MP4.box(MP4.types.tkhd, new Uint8Array([
      1,
      // version 1
      0,
      0,
      7,
      // flags
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      2,
      // creation_time
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      3,
      // modification_time
      id >> 24 & 255,
      id >> 16 & 255,
      id >> 8 & 255,
      id & 255,
      // track_ID
      0,
      0,
      0,
      0,
      // reserved
      upperWordDuration >> 24,
      upperWordDuration >> 16 & 255,
      upperWordDuration >> 8 & 255,
      upperWordDuration & 255,
      lowerWordDuration >> 24,
      lowerWordDuration >> 16 & 255,
      lowerWordDuration >> 8 & 255,
      lowerWordDuration & 255,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      // reserved
      0,
      0,
      // layer
      0,
      0,
      // alternate_group
      0,
      0,
      // non-audio track volume
      0,
      0,
      // reserved
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      64,
      0,
      0,
      0,
      // transformation: unity matrix
      width >> 8 & 255,
      width & 255,
      0,
      0,
      // width
      height >> 8 & 255,
      height & 255,
      0,
      0
      // height
    ]));
  }
  static traf(track, baseMediaDecodeTime) {
    const sampleDependencyTable = MP4.sdtp(track);
    const id = track.id;
    const upperWordBaseMediaDecodeTime = Math.floor(baseMediaDecodeTime / (UINT32_MAX + 1));
    const lowerWordBaseMediaDecodeTime = Math.floor(baseMediaDecodeTime % (UINT32_MAX + 1));
    return MP4.box(
      MP4.types.traf,
      MP4.box(MP4.types.tfhd, new Uint8Array([
        0,
        // version 0
        0,
        0,
        0,
        // flags
        id >> 24,
        id >> 16 & 255,
        id >> 8 & 255,
        id & 255
        // track_ID
      ])),
      MP4.box(MP4.types.tfdt, new Uint8Array([
        1,
        // version 1
        0,
        0,
        0,
        // flags
        upperWordBaseMediaDecodeTime >> 24,
        upperWordBaseMediaDecodeTime >> 16 & 255,
        upperWordBaseMediaDecodeTime >> 8 & 255,
        upperWordBaseMediaDecodeTime & 255,
        lowerWordBaseMediaDecodeTime >> 24,
        lowerWordBaseMediaDecodeTime >> 16 & 255,
        lowerWordBaseMediaDecodeTime >> 8 & 255,
        lowerWordBaseMediaDecodeTime & 255
      ])),
      MP4.trun(track, sampleDependencyTable.length + 16 + // tfhd
      20 + // tfdt
      8 + // traf header
      16 + // mfhd
      8 + // moof header
      8),
      // mdat header
      sampleDependencyTable
    );
  }
  /**
   * Generate a track box.
   * @param track a track definition
   */
  static trak(track) {
    track.duration = track.duration || 4294967295;
    return MP4.box(MP4.types.trak, MP4.tkhd(track), MP4.mdia(track));
  }
  static trex(track) {
    const id = track.id;
    return MP4.box(MP4.types.trex, new Uint8Array([
      0,
      // version 0
      0,
      0,
      0,
      // flags
      id >> 24,
      id >> 16 & 255,
      id >> 8 & 255,
      id & 255,
      // track_ID
      0,
      0,
      0,
      1,
      // default_sample_description_index
      0,
      0,
      0,
      0,
      // default_sample_duration
      0,
      0,
      0,
      0,
      // default_sample_size
      0,
      1,
      0,
      1
      // default_sample_flags
    ]));
  }
  static trun(track, offset) {
    const samples = track.samples || [];
    const len = samples.length;
    const arraylen = 12 + 16 * len;
    const array = new Uint8Array(arraylen);
    let i2;
    let sample;
    let duration;
    let size;
    let flags;
    let cts;
    offset += 8 + arraylen;
    array.set([
      track.type === "video" ? 1 : 0,
      // version 1 for video with signed-int sample_composition_time_offset
      0,
      15,
      1,
      // flags
      len >>> 24 & 255,
      len >>> 16 & 255,
      len >>> 8 & 255,
      len & 255,
      // sample_count
      offset >>> 24 & 255,
      offset >>> 16 & 255,
      offset >>> 8 & 255,
      offset & 255
      // data_offset
    ], 0);
    for (i2 = 0; i2 < len; i2++) {
      sample = samples[i2];
      duration = sample.duration;
      size = sample.size;
      flags = sample.flags;
      cts = sample.cts;
      array.set([
        duration >>> 24 & 255,
        duration >>> 16 & 255,
        duration >>> 8 & 255,
        duration & 255,
        // sample_duration
        size >>> 24 & 255,
        size >>> 16 & 255,
        size >>> 8 & 255,
        size & 255,
        // sample_size
        flags.isLeading << 2 | flags.dependsOn,
        flags.isDependedOn << 6 | flags.hasRedundancy << 4 | flags.paddingValue << 1 | flags.isNonSync,
        flags.degradPrio & 240 << 8,
        flags.degradPrio & 15,
        // sample_flags
        cts >>> 24 & 255,
        cts >>> 16 & 255,
        cts >>> 8 & 255,
        cts & 255
        // sample_composition_time_offset
      ], 12 + 16 * i2);
    }
    return MP4.box(MP4.types.trun, array);
  }
  static initSegment(tracks) {
    if (!MP4.types) {
      MP4.init();
    }
    const movie = MP4.moov(tracks);
    const result = appendUint8Array(MP4.FTYP, movie);
    return result;
  }
}
MP4.types = void 0;
MP4.HDLR_TYPES = void 0;
MP4.STTS = void 0;
MP4.STSC = void 0;
MP4.STCO = void 0;
MP4.STSZ = void 0;
MP4.VMHD = void 0;
MP4.SMHD = void 0;
MP4.STSD = void 0;
MP4.FTYP = void 0;
MP4.DINF = void 0;
const MPEG_TS_CLOCK_FREQ_HZ = 9e4;
function toTimescaleFromBase(baseTime, destScale, srcBase = 1, round = false) {
  const result = baseTime * destScale * srcBase;
  return round ? Math.round(result) : result;
}
function toTimescaleFromScale(baseTime, destScale, srcScale = 1, round = false) {
  return toTimescaleFromBase(baseTime, destScale, 1 / srcScale, round);
}
function toMsFromMpegTsClock(baseTime, round = false) {
  return toTimescaleFromBase(baseTime, 1e3, 1 / MPEG_TS_CLOCK_FREQ_HZ, round);
}
function toMpegTsClockFromTimescale(baseTime, srcScale = 1) {
  return toTimescaleFromBase(baseTime, MPEG_TS_CLOCK_FREQ_HZ, 1 / srcScale);
}
const MAX_SILENT_FRAME_DURATION = 10 * 1e3;
const AAC_SAMPLES_PER_FRAME = 1024;
const MPEG_AUDIO_SAMPLE_PER_FRAME = 1152;
const AC3_SAMPLES_PER_FRAME = 1536;
let chromeVersion = null;
let safariWebkitVersion = null;
class MP4Remuxer {
  constructor(observer, config, typeSupported, vendor = "") {
    this.observer = void 0;
    this.config = void 0;
    this.typeSupported = void 0;
    this.ISGenerated = false;
    this._initPTS = null;
    this._initDTS = null;
    this.nextAvcDts = null;
    this.nextAudioPts = null;
    this.videoSampleDuration = null;
    this.isAudioContiguous = false;
    this.isVideoContiguous = false;
    this.videoTrackConfig = void 0;
    this.observer = observer;
    this.config = config;
    this.typeSupported = typeSupported;
    this.ISGenerated = false;
    if (chromeVersion === null) {
      const userAgent = navigator.userAgent || "";
      const result = userAgent.match(/Chrome\/(\d+)/i);
      chromeVersion = result ? parseInt(result[1]) : 0;
    }
    if (safariWebkitVersion === null) {
      const result = navigator.userAgent.match(/Safari\/(\d+)/i);
      safariWebkitVersion = result ? parseInt(result[1]) : 0;
    }
  }
  destroy() {
    this.config = this.videoTrackConfig = this._initPTS = this._initDTS = null;
  }
  resetTimeStamp(defaultTimeStamp) {
    logger.log("[mp4-remuxer]: initPTS & initDTS reset");
    this._initPTS = this._initDTS = defaultTimeStamp;
  }
  resetNextTimestamp() {
    logger.log("[mp4-remuxer]: reset next timestamp");
    this.isVideoContiguous = false;
    this.isAudioContiguous = false;
  }
  resetInitSegment() {
    logger.log("[mp4-remuxer]: ISGenerated flag reset");
    this.ISGenerated = false;
    this.videoTrackConfig = void 0;
  }
  getVideoStartPts(videoSamples) {
    let rolloverDetected = false;
    const startPTS = videoSamples.reduce((minPTS, sample) => {
      const delta = sample.pts - minPTS;
      if (delta < -4294967296) {
        rolloverDetected = true;
        return normalizePts(minPTS, sample.pts);
      } else if (delta > 0) {
        return minPTS;
      } else {
        return sample.pts;
      }
    }, videoSamples[0].pts);
    if (rolloverDetected) {
      logger.debug("PTS rollover detected");
    }
    return startPTS;
  }
  remux(audioTrack, videoTrack, id3Track, textTrack, timeOffset, accurateTimeOffset, flush, playlistType) {
    let video;
    let audio;
    let initSegment;
    let text;
    let id3;
    let independent;
    let audioTimeOffset = timeOffset;
    let videoTimeOffset = timeOffset;
    const hasAudio = audioTrack.pid > -1;
    const hasVideo = videoTrack.pid > -1;
    const length = videoTrack.samples.length;
    const enoughAudioSamples = audioTrack.samples.length > 0;
    const enoughVideoSamples = flush && length > 0 || length > 1;
    const canRemuxAvc = (!hasAudio || enoughAudioSamples) && (!hasVideo || enoughVideoSamples) || this.ISGenerated || flush;
    if (canRemuxAvc) {
      if (this.ISGenerated) {
        var _videoTrack$pixelRati, _config$pixelRatio, _videoTrack$pixelRati2, _config$pixelRatio2;
        const config = this.videoTrackConfig;
        if (config && (videoTrack.width !== config.width || videoTrack.height !== config.height || ((_videoTrack$pixelRati = videoTrack.pixelRatio) == null ? void 0 : _videoTrack$pixelRati[0]) !== ((_config$pixelRatio = config.pixelRatio) == null ? void 0 : _config$pixelRatio[0]) || ((_videoTrack$pixelRati2 = videoTrack.pixelRatio) == null ? void 0 : _videoTrack$pixelRati2[1]) !== ((_config$pixelRatio2 = config.pixelRatio) == null ? void 0 : _config$pixelRatio2[1]))) {
          this.resetInitSegment();
        }
      } else {
        initSegment = this.generateIS(audioTrack, videoTrack, timeOffset, accurateTimeOffset);
      }
      const isVideoContiguous = this.isVideoContiguous;
      let firstKeyFrameIndex = -1;
      let firstKeyFramePTS;
      if (enoughVideoSamples) {
        firstKeyFrameIndex = findKeyframeIndex(videoTrack.samples);
        if (!isVideoContiguous && this.config.forceKeyFrameOnDiscontinuity) {
          independent = true;
          if (firstKeyFrameIndex > 0) {
            logger.warn(`[mp4-remuxer]: Dropped ${firstKeyFrameIndex} out of ${length} video samples due to a missing keyframe`);
            const startPTS = this.getVideoStartPts(videoTrack.samples);
            videoTrack.samples = videoTrack.samples.slice(firstKeyFrameIndex);
            videoTrack.dropped += firstKeyFrameIndex;
            videoTimeOffset += (videoTrack.samples[0].pts - startPTS) / videoTrack.inputTimeScale;
            firstKeyFramePTS = videoTimeOffset;
          } else if (firstKeyFrameIndex === -1) {
            logger.warn(`[mp4-remuxer]: No keyframe found out of ${length} video samples`);
            independent = false;
          }
        }
      }
      if (this.ISGenerated) {
        if (enoughAudioSamples && enoughVideoSamples) {
          const startPTS = this.getVideoStartPts(videoTrack.samples);
          const tsDelta = normalizePts(audioTrack.samples[0].pts, startPTS) - startPTS;
          const audiovideoTimestampDelta = tsDelta / videoTrack.inputTimeScale;
          audioTimeOffset += Math.max(0, audiovideoTimestampDelta);
          videoTimeOffset += Math.max(0, -audiovideoTimestampDelta);
        }
        if (enoughAudioSamples) {
          if (!audioTrack.samplerate) {
            logger.warn("[mp4-remuxer]: regenerate InitSegment as audio detected");
            initSegment = this.generateIS(audioTrack, videoTrack, timeOffset, accurateTimeOffset);
          }
          audio = this.remuxAudio(audioTrack, audioTimeOffset, this.isAudioContiguous, accurateTimeOffset, hasVideo || enoughVideoSamples || playlistType === PlaylistLevelType.AUDIO ? videoTimeOffset : void 0);
          if (enoughVideoSamples) {
            const audioTrackLength = audio ? audio.endPTS - audio.startPTS : 0;
            if (!videoTrack.inputTimeScale) {
              logger.warn("[mp4-remuxer]: regenerate InitSegment as video detected");
              initSegment = this.generateIS(audioTrack, videoTrack, timeOffset, accurateTimeOffset);
            }
            video = this.remuxVideo(videoTrack, videoTimeOffset, isVideoContiguous, audioTrackLength);
          }
        } else if (enoughVideoSamples) {
          video = this.remuxVideo(videoTrack, videoTimeOffset, isVideoContiguous, 0);
        }
        if (video) {
          video.firstKeyFrame = firstKeyFrameIndex;
          video.independent = firstKeyFrameIndex !== -1;
          video.firstKeyFramePTS = firstKeyFramePTS;
        }
      }
    }
    if (this.ISGenerated && this._initPTS && this._initDTS) {
      if (id3Track.samples.length) {
        id3 = flushTextTrackMetadataCueSamples(id3Track, timeOffset, this._initPTS, this._initDTS);
      }
      if (textTrack.samples.length) {
        text = flushTextTrackUserdataCueSamples(textTrack, timeOffset, this._initPTS);
      }
    }
    return {
      audio,
      video,
      initSegment,
      independent,
      text,
      id3
    };
  }
  generateIS(audioTrack, videoTrack, timeOffset, accurateTimeOffset) {
    const audioSamples = audioTrack.samples;
    const videoSamples = videoTrack.samples;
    const typeSupported = this.typeSupported;
    const tracks = {};
    const _initPTS = this._initPTS;
    let computePTSDTS = !_initPTS || accurateTimeOffset;
    let container = "audio/mp4";
    let initPTS;
    let initDTS;
    let timescale;
    if (computePTSDTS) {
      initPTS = initDTS = Infinity;
    }
    if (audioTrack.config && audioSamples.length) {
      audioTrack.timescale = audioTrack.samplerate;
      switch (audioTrack.segmentCodec) {
        case "mp3":
          if (typeSupported.mpeg) {
            container = "audio/mpeg";
            audioTrack.codec = "";
          } else if (typeSupported.mp3) {
            audioTrack.codec = "mp3";
          }
          break;
        case "ac3":
          audioTrack.codec = "ac-3";
          break;
      }
      tracks.audio = {
        id: "audio",
        container,
        codec: audioTrack.codec,
        initSegment: audioTrack.segmentCodec === "mp3" && typeSupported.mpeg ? new Uint8Array(0) : MP4.initSegment([audioTrack]),
        metadata: {
          channelCount: audioTrack.channelCount
        }
      };
      if (computePTSDTS) {
        timescale = audioTrack.inputTimeScale;
        if (!_initPTS || timescale !== _initPTS.timescale) {
          initPTS = initDTS = audioSamples[0].pts - Math.round(timescale * timeOffset);
        } else {
          computePTSDTS = false;
        }
      }
    }
    if (videoTrack.sps && videoTrack.pps && videoSamples.length) {
      videoTrack.timescale = videoTrack.inputTimeScale;
      tracks.video = {
        id: "main",
        container: "video/mp4",
        codec: videoTrack.codec,
        initSegment: MP4.initSegment([videoTrack]),
        metadata: {
          width: videoTrack.width,
          height: videoTrack.height
        }
      };
      if (computePTSDTS) {
        timescale = videoTrack.inputTimeScale;
        if (!_initPTS || timescale !== _initPTS.timescale) {
          const startPTS = this.getVideoStartPts(videoSamples);
          const startOffset = Math.round(timescale * timeOffset);
          initDTS = Math.min(initDTS, normalizePts(videoSamples[0].dts, startPTS) - startOffset);
          initPTS = Math.min(initPTS, startPTS - startOffset);
        } else {
          computePTSDTS = false;
        }
      }
      this.videoTrackConfig = {
        width: videoTrack.width,
        height: videoTrack.height,
        pixelRatio: videoTrack.pixelRatio
      };
    }
    if (Object.keys(tracks).length) {
      this.ISGenerated = true;
      if (computePTSDTS) {
        this._initPTS = {
          baseTime: initPTS,
          timescale
        };
        this._initDTS = {
          baseTime: initDTS,
          timescale
        };
      } else {
        initPTS = timescale = void 0;
      }
      return {
        tracks,
        initPTS,
        timescale
      };
    }
  }
  remuxVideo(track, timeOffset, contiguous, audioTrackLength) {
    const timeScale = track.inputTimeScale;
    const inputSamples = track.samples;
    const outputSamples = [];
    const nbSamples = inputSamples.length;
    const initPTS = this._initPTS;
    let nextAvcDts = this.nextAvcDts;
    let offset = 8;
    let mp4SampleDuration = this.videoSampleDuration;
    let firstDTS;
    let lastDTS;
    let minPTS = Number.POSITIVE_INFINITY;
    let maxPTS = Number.NEGATIVE_INFINITY;
    let sortSamples = false;
    if (!contiguous || nextAvcDts === null) {
      const pts = timeOffset * timeScale;
      const cts = inputSamples[0].pts - normalizePts(inputSamples[0].dts, inputSamples[0].pts);
      if (chromeVersion && nextAvcDts !== null && Math.abs(pts - cts - nextAvcDts) < 15e3) {
        contiguous = true;
      } else {
        nextAvcDts = pts - cts;
      }
    }
    const initTime = initPTS.baseTime * timeScale / initPTS.timescale;
    for (let i2 = 0; i2 < nbSamples; i2++) {
      const sample = inputSamples[i2];
      sample.pts = normalizePts(sample.pts - initTime, nextAvcDts);
      sample.dts = normalizePts(sample.dts - initTime, nextAvcDts);
      if (sample.dts < inputSamples[i2 > 0 ? i2 - 1 : i2].dts) {
        sortSamples = true;
      }
    }
    if (sortSamples) {
      inputSamples.sort(function(a, b) {
        const deltadts = a.dts - b.dts;
        const deltapts = a.pts - b.pts;
        return deltadts || deltapts;
      });
    }
    firstDTS = inputSamples[0].dts;
    lastDTS = inputSamples[inputSamples.length - 1].dts;
    const inputDuration = lastDTS - firstDTS;
    const averageSampleDuration = inputDuration ? Math.round(inputDuration / (nbSamples - 1)) : mp4SampleDuration || track.inputTimeScale / 30;
    if (contiguous) {
      const delta = firstDTS - nextAvcDts;
      const foundHole = delta > averageSampleDuration;
      const foundOverlap = delta < -1;
      if (foundHole || foundOverlap) {
        if (foundHole) {
          logger.warn(`AVC: ${toMsFromMpegTsClock(delta, true)} ms (${delta}dts) hole between fragments detected at ${timeOffset.toFixed(3)}`);
        } else {
          logger.warn(`AVC: ${toMsFromMpegTsClock(-delta, true)} ms (${delta}dts) overlapping between fragments detected at ${timeOffset.toFixed(3)}`);
        }
        if (!foundOverlap || nextAvcDts >= inputSamples[0].pts || chromeVersion) {
          firstDTS = nextAvcDts;
          const firstPTS = inputSamples[0].pts - delta;
          if (foundHole) {
            inputSamples[0].dts = firstDTS;
            inputSamples[0].pts = firstPTS;
          } else {
            for (let i2 = 0; i2 < inputSamples.length; i2++) {
              if (inputSamples[i2].dts > firstPTS) {
                break;
              }
              inputSamples[i2].dts -= delta;
              inputSamples[i2].pts -= delta;
            }
          }
          logger.log(`Video: Initial PTS/DTS adjusted: ${toMsFromMpegTsClock(firstPTS, true)}/${toMsFromMpegTsClock(firstDTS, true)}, delta: ${toMsFromMpegTsClock(delta, true)} ms`);
        }
      }
    }
    firstDTS = Math.max(0, firstDTS);
    let nbNalu = 0;
    let naluLen = 0;
    let dtsStep = firstDTS;
    for (let i2 = 0; i2 < nbSamples; i2++) {
      const sample = inputSamples[i2];
      const units = sample.units;
      const nbUnits = units.length;
      let sampleLen = 0;
      for (let j = 0; j < nbUnits; j++) {
        sampleLen += units[j].data.length;
      }
      naluLen += sampleLen;
      nbNalu += nbUnits;
      sample.length = sampleLen;
      if (sample.dts < dtsStep) {
        sample.dts = dtsStep;
        dtsStep += averageSampleDuration / 4 | 0 || 1;
      } else {
        dtsStep = sample.dts;
      }
      minPTS = Math.min(sample.pts, minPTS);
      maxPTS = Math.max(sample.pts, maxPTS);
    }
    lastDTS = inputSamples[nbSamples - 1].dts;
    const mdatSize = naluLen + 4 * nbNalu + 8;
    let mdat;
    try {
      mdat = new Uint8Array(mdatSize);
    } catch (err) {
      this.observer.emit(Events.ERROR, Events.ERROR, {
        type: ErrorTypes.MUX_ERROR,
        details: ErrorDetails.REMUX_ALLOC_ERROR,
        fatal: false,
        error: err,
        bytes: mdatSize,
        reason: `fail allocating video mdat ${mdatSize}`
      });
      return;
    }
    const view = new DataView(mdat.buffer);
    view.setUint32(0, mdatSize);
    mdat.set(MP4.types.mdat, 4);
    let stretchedLastFrame = false;
    let minDtsDelta = Number.POSITIVE_INFINITY;
    let minPtsDelta = Number.POSITIVE_INFINITY;
    let maxDtsDelta = Number.NEGATIVE_INFINITY;
    let maxPtsDelta = Number.NEGATIVE_INFINITY;
    for (let i2 = 0; i2 < nbSamples; i2++) {
      const VideoSample = inputSamples[i2];
      const VideoSampleUnits = VideoSample.units;
      let mp4SampleLength = 0;
      for (let j = 0, nbUnits = VideoSampleUnits.length; j < nbUnits; j++) {
        const unit = VideoSampleUnits[j];
        const unitData = unit.data;
        const unitDataLen = unit.data.byteLength;
        view.setUint32(offset, unitDataLen);
        offset += 4;
        mdat.set(unitData, offset);
        offset += unitDataLen;
        mp4SampleLength += 4 + unitDataLen;
      }
      let ptsDelta;
      if (i2 < nbSamples - 1) {
        mp4SampleDuration = inputSamples[i2 + 1].dts - VideoSample.dts;
        ptsDelta = inputSamples[i2 + 1].pts - VideoSample.pts;
      } else {
        const config = this.config;
        const lastFrameDuration = i2 > 0 ? VideoSample.dts - inputSamples[i2 - 1].dts : averageSampleDuration;
        ptsDelta = i2 > 0 ? VideoSample.pts - inputSamples[i2 - 1].pts : averageSampleDuration;
        if (config.stretchShortVideoTrack && this.nextAudioPts !== null) {
          const gapTolerance = Math.floor(config.maxBufferHole * timeScale);
          const deltaToFrameEnd = (audioTrackLength ? minPTS + audioTrackLength * timeScale : this.nextAudioPts) - VideoSample.pts;
          if (deltaToFrameEnd > gapTolerance) {
            mp4SampleDuration = deltaToFrameEnd - lastFrameDuration;
            if (mp4SampleDuration < 0) {
              mp4SampleDuration = lastFrameDuration;
            } else {
              stretchedLastFrame = true;
            }
            logger.log(`[mp4-remuxer]: It is approximately ${deltaToFrameEnd / 90} ms to the next segment; using duration ${mp4SampleDuration / 90} ms for the last video frame.`);
          } else {
            mp4SampleDuration = lastFrameDuration;
          }
        } else {
          mp4SampleDuration = lastFrameDuration;
        }
      }
      const compositionTimeOffset = Math.round(VideoSample.pts - VideoSample.dts);
      minDtsDelta = Math.min(minDtsDelta, mp4SampleDuration);
      maxDtsDelta = Math.max(maxDtsDelta, mp4SampleDuration);
      minPtsDelta = Math.min(minPtsDelta, ptsDelta);
      maxPtsDelta = Math.max(maxPtsDelta, ptsDelta);
      outputSamples.push(new Mp4Sample(VideoSample.key, mp4SampleDuration, mp4SampleLength, compositionTimeOffset));
    }
    if (outputSamples.length) {
      if (chromeVersion) {
        if (chromeVersion < 70) {
          const flags = outputSamples[0].flags;
          flags.dependsOn = 2;
          flags.isNonSync = 0;
        }
      } else if (safariWebkitVersion) {
        if (maxPtsDelta - minPtsDelta < maxDtsDelta - minDtsDelta && averageSampleDuration / maxDtsDelta < 0.025 && outputSamples[0].cts === 0) {
          logger.warn("Found irregular gaps in sample duration. Using PTS instead of DTS to determine MP4 sample duration.");
          let dts = firstDTS;
          for (let i2 = 0, len = outputSamples.length; i2 < len; i2++) {
            const nextDts = dts + outputSamples[i2].duration;
            const pts = dts + outputSamples[i2].cts;
            if (i2 < len - 1) {
              const nextPts = nextDts + outputSamples[i2 + 1].cts;
              outputSamples[i2].duration = nextPts - pts;
            } else {
              outputSamples[i2].duration = i2 ? outputSamples[i2 - 1].duration : averageSampleDuration;
            }
            outputSamples[i2].cts = 0;
            dts = nextDts;
          }
        }
      }
    }
    mp4SampleDuration = stretchedLastFrame || !mp4SampleDuration ? averageSampleDuration : mp4SampleDuration;
    this.nextAvcDts = nextAvcDts = lastDTS + mp4SampleDuration;
    this.videoSampleDuration = mp4SampleDuration;
    this.isVideoContiguous = true;
    const moof = MP4.moof(track.sequenceNumber++, firstDTS, _extends({}, track, {
      samples: outputSamples
    }));
    const type = "video";
    const data = {
      data1: moof,
      data2: mdat,
      startPTS: minPTS / timeScale,
      endPTS: (maxPTS + mp4SampleDuration) / timeScale,
      startDTS: firstDTS / timeScale,
      endDTS: nextAvcDts / timeScale,
      type,
      hasAudio: false,
      hasVideo: true,
      nb: outputSamples.length,
      dropped: track.dropped
    };
    track.samples = [];
    track.dropped = 0;
    return data;
  }
  getSamplesPerFrame(track) {
    switch (track.segmentCodec) {
      case "mp3":
        return MPEG_AUDIO_SAMPLE_PER_FRAME;
      case "ac3":
        return AC3_SAMPLES_PER_FRAME;
      default:
        return AAC_SAMPLES_PER_FRAME;
    }
  }
  remuxAudio(track, timeOffset, contiguous, accurateTimeOffset, videoTimeOffset) {
    const inputTimeScale = track.inputTimeScale;
    const mp4timeScale = track.samplerate ? track.samplerate : inputTimeScale;
    const scaleFactor = inputTimeScale / mp4timeScale;
    const mp4SampleDuration = this.getSamplesPerFrame(track);
    const inputSampleDuration = mp4SampleDuration * scaleFactor;
    const initPTS = this._initPTS;
    const rawMPEG = track.segmentCodec === "mp3" && this.typeSupported.mpeg;
    const outputSamples = [];
    const alignedWithVideo = videoTimeOffset !== void 0;
    let inputSamples = track.samples;
    let offset = rawMPEG ? 0 : 8;
    let nextAudioPts = this.nextAudioPts || -1;
    const timeOffsetMpegTS = timeOffset * inputTimeScale;
    const initTime = initPTS.baseTime * inputTimeScale / initPTS.timescale;
    this.isAudioContiguous = contiguous = contiguous || inputSamples.length && nextAudioPts > 0 && (accurateTimeOffset && Math.abs(timeOffsetMpegTS - nextAudioPts) < 9e3 || Math.abs(normalizePts(inputSamples[0].pts - initTime, timeOffsetMpegTS) - nextAudioPts) < 20 * inputSampleDuration);
    inputSamples.forEach(function(sample) {
      sample.pts = normalizePts(sample.pts - initTime, timeOffsetMpegTS);
    });
    if (!contiguous || nextAudioPts < 0) {
      inputSamples = inputSamples.filter((sample) => sample.pts >= 0);
      if (!inputSamples.length) {
        return;
      }
      if (videoTimeOffset === 0) {
        nextAudioPts = 0;
      } else if (accurateTimeOffset && !alignedWithVideo) {
        nextAudioPts = Math.max(0, timeOffsetMpegTS);
      } else {
        nextAudioPts = inputSamples[0].pts;
      }
    }
    if (track.segmentCodec === "aac") {
      const maxAudioFramesDrift = this.config.maxAudioFramesDrift;
      for (let i2 = 0, nextPts = nextAudioPts; i2 < inputSamples.length; i2++) {
        const sample = inputSamples[i2];
        const pts = sample.pts;
        const delta = pts - nextPts;
        const duration = Math.abs(1e3 * delta / inputTimeScale);
        if (delta <= -maxAudioFramesDrift * inputSampleDuration && alignedWithVideo) {
          if (i2 === 0) {
            logger.warn(`Audio frame @ ${(pts / inputTimeScale).toFixed(3)}s overlaps nextAudioPts by ${Math.round(1e3 * delta / inputTimeScale)} ms.`);
            this.nextAudioPts = nextAudioPts = nextPts = pts;
          }
        } else if (delta >= maxAudioFramesDrift * inputSampleDuration && duration < MAX_SILENT_FRAME_DURATION && alignedWithVideo) {
          let missing = Math.round(delta / inputSampleDuration);
          nextPts = pts - missing * inputSampleDuration;
          if (nextPts < 0) {
            missing--;
            nextPts += inputSampleDuration;
          }
          if (i2 === 0) {
            this.nextAudioPts = nextAudioPts = nextPts;
          }
          logger.warn(`[mp4-remuxer]: Injecting ${missing} audio frame @ ${(nextPts / inputTimeScale).toFixed(3)}s due to ${Math.round(1e3 * delta / inputTimeScale)} ms gap.`);
          for (let j = 0; j < missing; j++) {
            const newStamp = Math.max(nextPts, 0);
            let fillFrame = AAC.getSilentFrame(track.manifestCodec || track.codec, track.channelCount);
            if (!fillFrame) {
              logger.log("[mp4-remuxer]: Unable to get silent frame for given audio codec; duplicating last frame instead.");
              fillFrame = sample.unit.subarray();
            }
            inputSamples.splice(i2, 0, {
              unit: fillFrame,
              pts: newStamp
            });
            nextPts += inputSampleDuration;
            i2++;
          }
        }
        sample.pts = nextPts;
        nextPts += inputSampleDuration;
      }
    }
    let firstPTS = null;
    let lastPTS = null;
    let mdat;
    let mdatSize = 0;
    let sampleLength = inputSamples.length;
    while (sampleLength--) {
      mdatSize += inputSamples[sampleLength].unit.byteLength;
    }
    for (let j = 0, _nbSamples = inputSamples.length; j < _nbSamples; j++) {
      const audioSample = inputSamples[j];
      const unit = audioSample.unit;
      let pts = audioSample.pts;
      if (lastPTS !== null) {
        const prevSample = outputSamples[j - 1];
        prevSample.duration = Math.round((pts - lastPTS) / scaleFactor);
      } else {
        if (contiguous && track.segmentCodec === "aac") {
          pts = nextAudioPts;
        }
        firstPTS = pts;
        if (mdatSize > 0) {
          mdatSize += offset;
          try {
            mdat = new Uint8Array(mdatSize);
          } catch (err) {
            this.observer.emit(Events.ERROR, Events.ERROR, {
              type: ErrorTypes.MUX_ERROR,
              details: ErrorDetails.REMUX_ALLOC_ERROR,
              fatal: false,
              error: err,
              bytes: mdatSize,
              reason: `fail allocating audio mdat ${mdatSize}`
            });
            return;
          }
          if (!rawMPEG) {
            const view = new DataView(mdat.buffer);
            view.setUint32(0, mdatSize);
            mdat.set(MP4.types.mdat, 4);
          }
        } else {
          return;
        }
      }
      mdat.set(unit, offset);
      const unitLen = unit.byteLength;
      offset += unitLen;
      outputSamples.push(new Mp4Sample(true, mp4SampleDuration, unitLen, 0));
      lastPTS = pts;
    }
    const nbSamples = outputSamples.length;
    if (!nbSamples) {
      return;
    }
    const lastSample = outputSamples[outputSamples.length - 1];
    this.nextAudioPts = nextAudioPts = lastPTS + scaleFactor * lastSample.duration;
    const moof = rawMPEG ? new Uint8Array(0) : MP4.moof(track.sequenceNumber++, firstPTS / scaleFactor, _extends({}, track, {
      samples: outputSamples
    }));
    track.samples = [];
    const start = firstPTS / inputTimeScale;
    const end = nextAudioPts / inputTimeScale;
    const type = "audio";
    const audioData = {
      data1: moof,
      data2: mdat,
      startPTS: start,
      endPTS: end,
      startDTS: start,
      endDTS: end,
      type,
      hasAudio: true,
      hasVideo: false,
      nb: nbSamples
    };
    this.isAudioContiguous = true;
    return audioData;
  }
  remuxEmptyAudio(track, timeOffset, contiguous, videoData) {
    const inputTimeScale = track.inputTimeScale;
    const mp4timeScale = track.samplerate ? track.samplerate : inputTimeScale;
    const scaleFactor = inputTimeScale / mp4timeScale;
    const nextAudioPts = this.nextAudioPts;
    const initDTS = this._initDTS;
    const init90kHz = initDTS.baseTime * 9e4 / initDTS.timescale;
    const startDTS = (nextAudioPts !== null ? nextAudioPts : videoData.startDTS * inputTimeScale) + init90kHz;
    const endDTS = videoData.endDTS * inputTimeScale + init90kHz;
    const frameDuration = scaleFactor * AAC_SAMPLES_PER_FRAME;
    const nbSamples = Math.ceil((endDTS - startDTS) / frameDuration);
    const silentFrame = AAC.getSilentFrame(track.manifestCodec || track.codec, track.channelCount);
    logger.warn("[mp4-remuxer]: remux empty Audio");
    if (!silentFrame) {
      logger.trace("[mp4-remuxer]: Unable to remuxEmptyAudio since we were unable to get a silent frame for given audio codec");
      return;
    }
    const samples = [];
    for (let i2 = 0; i2 < nbSamples; i2++) {
      const stamp = startDTS + i2 * frameDuration;
      samples.push({
        unit: silentFrame,
        pts: stamp,
        dts: stamp
      });
    }
    track.samples = samples;
    return this.remuxAudio(track, timeOffset, contiguous, false);
  }
}
function normalizePts(value, reference) {
  let offset;
  if (reference === null) {
    return value;
  }
  if (reference < value) {
    offset = -8589934592;
  } else {
    offset = 8589934592;
  }
  while (Math.abs(value - reference) > 4294967296) {
    value += offset;
  }
  return value;
}
function findKeyframeIndex(samples) {
  for (let i2 = 0; i2 < samples.length; i2++) {
    if (samples[i2].key) {
      return i2;
    }
  }
  return -1;
}
function flushTextTrackMetadataCueSamples(track, timeOffset, initPTS, initDTS) {
  const length = track.samples.length;
  if (!length) {
    return;
  }
  const inputTimeScale = track.inputTimeScale;
  for (let index = 0; index < length; index++) {
    const sample = track.samples[index];
    sample.pts = normalizePts(sample.pts - initPTS.baseTime * inputTimeScale / initPTS.timescale, timeOffset * inputTimeScale) / inputTimeScale;
    sample.dts = normalizePts(sample.dts - initDTS.baseTime * inputTimeScale / initDTS.timescale, timeOffset * inputTimeScale) / inputTimeScale;
  }
  const samples = track.samples;
  track.samples = [];
  return {
    samples
  };
}
function flushTextTrackUserdataCueSamples(track, timeOffset, initPTS) {
  const length = track.samples.length;
  if (!length) {
    return;
  }
  const inputTimeScale = track.inputTimeScale;
  for (let index = 0; index < length; index++) {
    const sample = track.samples[index];
    sample.pts = normalizePts(sample.pts - initPTS.baseTime * inputTimeScale / initPTS.timescale, timeOffset * inputTimeScale) / inputTimeScale;
  }
  track.samples.sort((a, b) => a.pts - b.pts);
  const samples = track.samples;
  track.samples = [];
  return {
    samples
  };
}
class Mp4Sample {
  constructor(isKeyframe, duration, size, cts) {
    this.size = void 0;
    this.duration = void 0;
    this.cts = void 0;
    this.flags = void 0;
    this.duration = duration;
    this.size = size;
    this.cts = cts;
    this.flags = {
      isLeading: 0,
      isDependedOn: 0,
      hasRedundancy: 0,
      degradPrio: 0,
      dependsOn: isKeyframe ? 2 : 1,
      isNonSync: isKeyframe ? 0 : 1
    };
  }
}
class PassThroughRemuxer {
  constructor() {
    this.emitInitSegment = false;
    this.audioCodec = void 0;
    this.videoCodec = void 0;
    this.initData = void 0;
    this.initPTS = null;
    this.initTracks = void 0;
    this.lastEndTime = null;
  }
  destroy() {
  }
  resetTimeStamp(defaultInitPTS) {
    this.initPTS = defaultInitPTS;
    this.lastEndTime = null;
  }
  resetNextTimestamp() {
    this.lastEndTime = null;
  }
  resetInitSegment(initSegment, audioCodec, videoCodec, decryptdata) {
    this.audioCodec = audioCodec;
    this.videoCodec = videoCodec;
    this.generateInitSegment(patchEncyptionData(initSegment, decryptdata));
    this.emitInitSegment = true;
  }
  generateInitSegment(initSegment) {
    let {
      audioCodec,
      videoCodec
    } = this;
    if (!(initSegment != null && initSegment.byteLength)) {
      this.initTracks = void 0;
      this.initData = void 0;
      return;
    }
    const initData = this.initData = parseInitSegment(initSegment);
    if (initData.audio) {
      audioCodec = getParsedTrackCodec(initData.audio, ElementaryStreamTypes.AUDIO);
    }
    if (initData.video) {
      videoCodec = getParsedTrackCodec(initData.video, ElementaryStreamTypes.VIDEO);
    }
    const tracks = {};
    if (initData.audio && initData.video) {
      tracks.audiovideo = {
        container: "video/mp4",
        codec: audioCodec + "," + videoCodec,
        initSegment,
        id: "main"
      };
    } else if (initData.audio) {
      tracks.audio = {
        container: "audio/mp4",
        codec: audioCodec,
        initSegment,
        id: "audio"
      };
    } else if (initData.video) {
      tracks.video = {
        container: "video/mp4",
        codec: videoCodec,
        initSegment,
        id: "main"
      };
    } else {
      logger.warn("[passthrough-remuxer.ts]: initSegment does not contain moov or trak boxes.");
    }
    this.initTracks = tracks;
  }
  remux(audioTrack, videoTrack, id3Track, textTrack, timeOffset, accurateTimeOffset) {
    var _initData, _initData2;
    let {
      initPTS,
      lastEndTime
    } = this;
    const result = {
      audio: void 0,
      video: void 0,
      text: textTrack,
      id3: id3Track,
      initSegment: void 0
    };
    if (!isFiniteNumber(lastEndTime)) {
      lastEndTime = this.lastEndTime = timeOffset || 0;
    }
    const data = videoTrack.samples;
    if (!(data != null && data.length)) {
      return result;
    }
    const initSegment = {
      initPTS: void 0,
      timescale: 1
    };
    let initData = this.initData;
    if (!((_initData = initData) != null && _initData.length)) {
      this.generateInitSegment(data);
      initData = this.initData;
    }
    if (!((_initData2 = initData) != null && _initData2.length)) {
      logger.warn("[passthrough-remuxer.ts]: Failed to generate initSegment.");
      return result;
    }
    if (this.emitInitSegment) {
      initSegment.tracks = this.initTracks;
      this.emitInitSegment = false;
    }
    const duration = getDuration(data, initData);
    const startDTS = getStartDTS(initData, data);
    const decodeTime = startDTS === null ? timeOffset : startDTS;
    if (isInvalidInitPts(initPTS, decodeTime, timeOffset, duration) || initSegment.timescale !== initPTS.timescale && accurateTimeOffset) {
      initSegment.initPTS = decodeTime - timeOffset;
      if (initPTS && initPTS.timescale === 1) {
        logger.warn(`Adjusting initPTS by ${initSegment.initPTS - initPTS.baseTime}`);
      }
      this.initPTS = initPTS = {
        baseTime: initSegment.initPTS,
        timescale: 1
      };
    }
    const startTime = audioTrack ? decodeTime - initPTS.baseTime / initPTS.timescale : lastEndTime;
    const endTime = startTime + duration;
    offsetStartDTS(initData, data, initPTS.baseTime / initPTS.timescale);
    if (duration > 0) {
      this.lastEndTime = endTime;
    } else {
      logger.warn("Duration parsed from mp4 should be greater than zero");
      this.resetNextTimestamp();
    }
    const hasAudio = !!initData.audio;
    const hasVideo = !!initData.video;
    let type = "";
    if (hasAudio) {
      type += "audio";
    }
    if (hasVideo) {
      type += "video";
    }
    const track = {
      data1: data,
      startPTS: startTime,
      startDTS: startTime,
      endPTS: endTime,
      endDTS: endTime,
      type,
      hasAudio,
      hasVideo,
      nb: 1,
      dropped: 0
    };
    result.audio = track.type === "audio" ? track : void 0;
    result.video = track.type !== "audio" ? track : void 0;
    result.initSegment = initSegment;
    result.id3 = flushTextTrackMetadataCueSamples(id3Track, timeOffset, initPTS, initPTS);
    if (textTrack.samples.length) {
      result.text = flushTextTrackUserdataCueSamples(textTrack, timeOffset, initPTS);
    }
    return result;
  }
}
function isInvalidInitPts(initPTS, startDTS, timeOffset, duration) {
  if (initPTS === null) {
    return true;
  }
  const minDuration = Math.max(duration, 1);
  const startTime = startDTS - initPTS.baseTime / initPTS.timescale;
  return Math.abs(startTime - timeOffset) > minDuration;
}
function getParsedTrackCodec(track, type) {
  const parsedCodec = track == null ? void 0 : track.codec;
  if (parsedCodec && parsedCodec.length > 4) {
    return parsedCodec;
  }
  if (type === ElementaryStreamTypes.AUDIO) {
    if (parsedCodec === "ec-3" || parsedCodec === "ac-3" || parsedCodec === "alac") {
      return parsedCodec;
    }
    if (parsedCodec === "fLaC" || parsedCodec === "Opus") {
      const preferManagedMediaSource = false;
      return getCodecCompatibleName(parsedCodec, preferManagedMediaSource);
    }
    const result = "mp4a.40.5";
    logger.info(`Parsed audio codec "${parsedCodec}" or audio object type not handled. Using "${result}"`);
    return result;
  }
  logger.warn(`Unhandled video codec "${parsedCodec}"`);
  if (parsedCodec === "hvc1" || parsedCodec === "hev1") {
    return "hvc1.1.6.L120.90";
  }
  if (parsedCodec === "av01") {
    return "av01.0.04M.08";
  }
  return "avc1.42e01e";
}
let now;
try {
  now = self.performance.now.bind(self.performance);
} catch (err) {
  logger.debug("Unable to use Performance API on this environment");
  now = optionalSelf == null ? void 0 : optionalSelf.Date.now;
}
const muxConfig = [{
  demux: MP4Demuxer,
  remux: PassThroughRemuxer
}, {
  demux: TSDemuxer,
  remux: MP4Remuxer
}, {
  demux: AACDemuxer,
  remux: MP4Remuxer
}, {
  demux: MP3Demuxer,
  remux: MP4Remuxer
}];
{
  muxConfig.splice(2, 0, {
    demux: AC3Demuxer,
    remux: MP4Remuxer
  });
}
class Transmuxer {
  constructor(observer, typeSupported, config, vendor, id) {
    this.async = false;
    this.observer = void 0;
    this.typeSupported = void 0;
    this.config = void 0;
    this.vendor = void 0;
    this.id = void 0;
    this.demuxer = void 0;
    this.remuxer = void 0;
    this.decrypter = void 0;
    this.probe = void 0;
    this.decryptionPromise = null;
    this.transmuxConfig = void 0;
    this.currentTransmuxState = void 0;
    this.observer = observer;
    this.typeSupported = typeSupported;
    this.config = config;
    this.vendor = vendor;
    this.id = id;
  }
  configure(transmuxConfig) {
    this.transmuxConfig = transmuxConfig;
    if (this.decrypter) {
      this.decrypter.reset();
    }
  }
  push(data, decryptdata, chunkMeta, state) {
    const stats = chunkMeta.transmuxing;
    stats.executeStart = now();
    let uintData = new Uint8Array(data);
    const {
      currentTransmuxState,
      transmuxConfig
    } = this;
    if (state) {
      this.currentTransmuxState = state;
    }
    const {
      contiguous,
      discontinuity,
      trackSwitch,
      accurateTimeOffset,
      timeOffset,
      initSegmentChange
    } = state || currentTransmuxState;
    const {
      audioCodec,
      videoCodec,
      defaultInitPts,
      duration,
      initSegmentData
    } = transmuxConfig;
    const keyData = getEncryptionType(uintData, decryptdata);
    if (keyData && keyData.method === "AES-128") {
      const decrypter = this.getDecrypter();
      if (decrypter.isSync()) {
        let decryptedData = decrypter.softwareDecrypt(uintData, keyData.key.buffer, keyData.iv.buffer);
        const loadingParts = chunkMeta.part > -1;
        if (loadingParts) {
          decryptedData = decrypter.flush();
        }
        if (!decryptedData) {
          stats.executeEnd = now();
          return emptyResult(chunkMeta);
        }
        uintData = new Uint8Array(decryptedData);
      } else {
        this.decryptionPromise = decrypter.webCryptoDecrypt(uintData, keyData.key.buffer, keyData.iv.buffer).then((decryptedData) => {
          const result2 = this.push(decryptedData, null, chunkMeta);
          this.decryptionPromise = null;
          return result2;
        });
        return this.decryptionPromise;
      }
    }
    const resetMuxers = this.needsProbing(discontinuity, trackSwitch);
    if (resetMuxers) {
      const error = this.configureTransmuxer(uintData);
      if (error) {
        logger.warn(`[transmuxer] ${error.message}`);
        this.observer.emit(Events.ERROR, Events.ERROR, {
          type: ErrorTypes.MEDIA_ERROR,
          details: ErrorDetails.FRAG_PARSING_ERROR,
          fatal: false,
          error,
          reason: error.message
        });
        stats.executeEnd = now();
        return emptyResult(chunkMeta);
      }
    }
    if (discontinuity || trackSwitch || initSegmentChange || resetMuxers) {
      this.resetInitSegment(initSegmentData, audioCodec, videoCodec, duration, decryptdata);
    }
    if (discontinuity || initSegmentChange || resetMuxers) {
      this.resetInitialTimestamp(defaultInitPts);
    }
    if (!contiguous) {
      this.resetContiguity();
    }
    const result = this.transmux(uintData, keyData, timeOffset, accurateTimeOffset, chunkMeta);
    const currentState = this.currentTransmuxState;
    currentState.contiguous = true;
    currentState.discontinuity = false;
    currentState.trackSwitch = false;
    stats.executeEnd = now();
    return result;
  }
  // Due to data caching, flush calls can produce more than one TransmuxerResult (hence the Array type)
  flush(chunkMeta) {
    const stats = chunkMeta.transmuxing;
    stats.executeStart = now();
    const {
      decrypter,
      currentTransmuxState,
      decryptionPromise
    } = this;
    if (decryptionPromise) {
      return decryptionPromise.then(() => {
        return this.flush(chunkMeta);
      });
    }
    const transmuxResults = [];
    const {
      timeOffset
    } = currentTransmuxState;
    if (decrypter) {
      const decryptedData = decrypter.flush();
      if (decryptedData) {
        transmuxResults.push(this.push(decryptedData, null, chunkMeta));
      }
    }
    const {
      demuxer,
      remuxer
    } = this;
    if (!demuxer || !remuxer) {
      stats.executeEnd = now();
      return [emptyResult(chunkMeta)];
    }
    const demuxResultOrPromise = demuxer.flush(timeOffset);
    if (isPromise(demuxResultOrPromise)) {
      return demuxResultOrPromise.then((demuxResult) => {
        this.flushRemux(transmuxResults, demuxResult, chunkMeta);
        return transmuxResults;
      });
    }
    this.flushRemux(transmuxResults, demuxResultOrPromise, chunkMeta);
    return transmuxResults;
  }
  flushRemux(transmuxResults, demuxResult, chunkMeta) {
    const {
      audioTrack,
      videoTrack,
      id3Track,
      textTrack
    } = demuxResult;
    const {
      accurateTimeOffset,
      timeOffset
    } = this.currentTransmuxState;
    logger.log(`[transmuxer.ts]: Flushed fragment ${chunkMeta.sn}${chunkMeta.part > -1 ? " p: " + chunkMeta.part : ""} of level ${chunkMeta.level}`);
    const remuxResult = this.remuxer.remux(audioTrack, videoTrack, id3Track, textTrack, timeOffset, accurateTimeOffset, true, this.id);
    transmuxResults.push({
      remuxResult,
      chunkMeta
    });
    chunkMeta.transmuxing.executeEnd = now();
  }
  resetInitialTimestamp(defaultInitPts) {
    const {
      demuxer,
      remuxer
    } = this;
    if (!demuxer || !remuxer) {
      return;
    }
    demuxer.resetTimeStamp(defaultInitPts);
    remuxer.resetTimeStamp(defaultInitPts);
  }
  resetContiguity() {
    const {
      demuxer,
      remuxer
    } = this;
    if (!demuxer || !remuxer) {
      return;
    }
    demuxer.resetContiguity();
    remuxer.resetNextTimestamp();
  }
  resetInitSegment(initSegmentData, audioCodec, videoCodec, trackDuration, decryptdata) {
    const {
      demuxer,
      remuxer
    } = this;
    if (!demuxer || !remuxer) {
      return;
    }
    demuxer.resetInitSegment(initSegmentData, audioCodec, videoCodec, trackDuration);
    remuxer.resetInitSegment(initSegmentData, audioCodec, videoCodec, decryptdata);
  }
  destroy() {
    if (this.demuxer) {
      this.demuxer.destroy();
      this.demuxer = void 0;
    }
    if (this.remuxer) {
      this.remuxer.destroy();
      this.remuxer = void 0;
    }
  }
  transmux(data, keyData, timeOffset, accurateTimeOffset, chunkMeta) {
    let result;
    if (keyData && keyData.method === "SAMPLE-AES") {
      result = this.transmuxSampleAes(data, keyData, timeOffset, accurateTimeOffset, chunkMeta);
    } else {
      result = this.transmuxUnencrypted(data, timeOffset, accurateTimeOffset, chunkMeta);
    }
    return result;
  }
  transmuxUnencrypted(data, timeOffset, accurateTimeOffset, chunkMeta) {
    const {
      audioTrack,
      videoTrack,
      id3Track,
      textTrack
    } = this.demuxer.demux(data, timeOffset, false, !this.config.progressive);
    const remuxResult = this.remuxer.remux(audioTrack, videoTrack, id3Track, textTrack, timeOffset, accurateTimeOffset, false, this.id);
    return {
      remuxResult,
      chunkMeta
    };
  }
  transmuxSampleAes(data, decryptData, timeOffset, accurateTimeOffset, chunkMeta) {
    return this.demuxer.demuxSampleAes(data, decryptData, timeOffset).then((demuxResult) => {
      const remuxResult = this.remuxer.remux(demuxResult.audioTrack, demuxResult.videoTrack, demuxResult.id3Track, demuxResult.textTrack, timeOffset, accurateTimeOffset, false, this.id);
      return {
        remuxResult,
        chunkMeta
      };
    });
  }
  configureTransmuxer(data) {
    const {
      config,
      observer,
      typeSupported,
      vendor
    } = this;
    let mux;
    for (let i2 = 0, len = muxConfig.length; i2 < len; i2++) {
      var _muxConfig$i$demux;
      if ((_muxConfig$i$demux = muxConfig[i2].demux) != null && _muxConfig$i$demux.probe(data)) {
        mux = muxConfig[i2];
        break;
      }
    }
    if (!mux) {
      return new Error("Failed to find demuxer by probing fragment data");
    }
    const demuxer = this.demuxer;
    const remuxer = this.remuxer;
    const Remuxer = mux.remux;
    const Demuxer = mux.demux;
    if (!remuxer || !(remuxer instanceof Remuxer)) {
      this.remuxer = new Remuxer(observer, config, typeSupported, vendor);
    }
    if (!demuxer || !(demuxer instanceof Demuxer)) {
      this.demuxer = new Demuxer(observer, config, typeSupported);
      this.probe = Demuxer.probe;
    }
  }
  needsProbing(discontinuity, trackSwitch) {
    return !this.demuxer || !this.remuxer || discontinuity || trackSwitch;
  }
  getDecrypter() {
    let decrypter = this.decrypter;
    if (!decrypter) {
      decrypter = this.decrypter = new Decrypter(this.config);
    }
    return decrypter;
  }
}
function getEncryptionType(data, decryptData) {
  let encryptionType = null;
  if (data.byteLength > 0 && (decryptData == null ? void 0 : decryptData.key) != null && decryptData.iv !== null && decryptData.method != null) {
    encryptionType = decryptData;
  }
  return encryptionType;
}
const emptyResult = (chunkMeta) => ({
  remuxResult: {},
  chunkMeta
});
function isPromise(p) {
  return "then" in p && p.then instanceof Function;
}
class TransmuxConfig {
  constructor(audioCodec, videoCodec, initSegmentData, duration, defaultInitPts) {
    this.audioCodec = void 0;
    this.videoCodec = void 0;
    this.initSegmentData = void 0;
    this.duration = void 0;
    this.defaultInitPts = void 0;
    this.audioCodec = audioCodec;
    this.videoCodec = videoCodec;
    this.initSegmentData = initSegmentData;
    this.duration = duration;
    this.defaultInitPts = defaultInitPts || null;
  }
}
class TransmuxState {
  constructor(discontinuity, contiguous, accurateTimeOffset, trackSwitch, timeOffset, initSegmentChange) {
    this.discontinuity = void 0;
    this.contiguous = void 0;
    this.accurateTimeOffset = void 0;
    this.trackSwitch = void 0;
    this.timeOffset = void 0;
    this.initSegmentChange = void 0;
    this.discontinuity = discontinuity;
    this.contiguous = contiguous;
    this.accurateTimeOffset = accurateTimeOffset;
    this.trackSwitch = trackSwitch;
    this.timeOffset = timeOffset;
    this.initSegmentChange = initSegmentChange;
  }
}
var eventemitter3 = { exports: {} };
(function(module) {
  var has = Object.prototype.hasOwnProperty, prefix = "~";
  function Events2() {
  }
  if (Object.create) {
    Events2.prototype = /* @__PURE__ */ Object.create(null);
    if (!new Events2().__proto__) prefix = false;
  }
  function EE(fn, context, once) {
    this.fn = fn;
    this.context = context;
    this.once = once || false;
  }
  function addListener(emitter, event, fn, context, once) {
    if (typeof fn !== "function") {
      throw new TypeError("The listener must be a function");
    }
    var listener = new EE(fn, context || emitter, once), evt = prefix ? prefix + event : event;
    if (!emitter._events[evt]) emitter._events[evt] = listener, emitter._eventsCount++;
    else if (!emitter._events[evt].fn) emitter._events[evt].push(listener);
    else emitter._events[evt] = [emitter._events[evt], listener];
    return emitter;
  }
  function clearEvent(emitter, evt) {
    if (--emitter._eventsCount === 0) emitter._events = new Events2();
    else delete emitter._events[evt];
  }
  function EventEmitter2() {
    this._events = new Events2();
    this._eventsCount = 0;
  }
  EventEmitter2.prototype.eventNames = function eventNames() {
    var names = [], events, name;
    if (this._eventsCount === 0) return names;
    for (name in events = this._events) {
      if (has.call(events, name)) names.push(prefix ? name.slice(1) : name);
    }
    if (Object.getOwnPropertySymbols) {
      return names.concat(Object.getOwnPropertySymbols(events));
    }
    return names;
  };
  EventEmitter2.prototype.listeners = function listeners(event) {
    var evt = prefix ? prefix + event : event, handlers = this._events[evt];
    if (!handlers) return [];
    if (handlers.fn) return [handlers.fn];
    for (var i2 = 0, l = handlers.length, ee = new Array(l); i2 < l; i2++) {
      ee[i2] = handlers[i2].fn;
    }
    return ee;
  };
  EventEmitter2.prototype.listenerCount = function listenerCount(event) {
    var evt = prefix ? prefix + event : event, listeners = this._events[evt];
    if (!listeners) return 0;
    if (listeners.fn) return 1;
    return listeners.length;
  };
  EventEmitter2.prototype.emit = function emit(event, a1, a2, a3, a4, a5) {
    var evt = prefix ? prefix + event : event;
    if (!this._events[evt]) return false;
    var listeners = this._events[evt], len = arguments.length, args, i2;
    if (listeners.fn) {
      if (listeners.once) this.removeListener(event, listeners.fn, void 0, true);
      switch (len) {
        case 1:
          return listeners.fn.call(listeners.context), true;
        case 2:
          return listeners.fn.call(listeners.context, a1), true;
        case 3:
          return listeners.fn.call(listeners.context, a1, a2), true;
        case 4:
          return listeners.fn.call(listeners.context, a1, a2, a3), true;
        case 5:
          return listeners.fn.call(listeners.context, a1, a2, a3, a4), true;
        case 6:
          return listeners.fn.call(listeners.context, a1, a2, a3, a4, a5), true;
      }
      for (i2 = 1, args = new Array(len - 1); i2 < len; i2++) {
        args[i2 - 1] = arguments[i2];
      }
      listeners.fn.apply(listeners.context, args);
    } else {
      var length = listeners.length, j;
      for (i2 = 0; i2 < length; i2++) {
        if (listeners[i2].once) this.removeListener(event, listeners[i2].fn, void 0, true);
        switch (len) {
          case 1:
            listeners[i2].fn.call(listeners[i2].context);
            break;
          case 2:
            listeners[i2].fn.call(listeners[i2].context, a1);
            break;
          case 3:
            listeners[i2].fn.call(listeners[i2].context, a1, a2);
            break;
          case 4:
            listeners[i2].fn.call(listeners[i2].context, a1, a2, a3);
            break;
          default:
            if (!args) for (j = 1, args = new Array(len - 1); j < len; j++) {
              args[j - 1] = arguments[j];
            }
            listeners[i2].fn.apply(listeners[i2].context, args);
        }
      }
    }
    return true;
  };
  EventEmitter2.prototype.on = function on(event, fn, context) {
    return addListener(this, event, fn, context, false);
  };
  EventEmitter2.prototype.once = function once(event, fn, context) {
    return addListener(this, event, fn, context, true);
  };
  EventEmitter2.prototype.removeListener = function removeListener(event, fn, context, once) {
    var evt = prefix ? prefix + event : event;
    if (!this._events[evt]) return this;
    if (!fn) {
      clearEvent(this, evt);
      return this;
    }
    var listeners = this._events[evt];
    if (listeners.fn) {
      if (listeners.fn === fn && (!once || listeners.once) && (!context || listeners.context === context)) {
        clearEvent(this, evt);
      }
    } else {
      for (var i2 = 0, events = [], length = listeners.length; i2 < length; i2++) {
        if (listeners[i2].fn !== fn || once && !listeners[i2].once || context && listeners[i2].context !== context) {
          events.push(listeners[i2]);
        }
      }
      if (events.length) this._events[evt] = events.length === 1 ? events[0] : events;
      else clearEvent(this, evt);
    }
    return this;
  };
  EventEmitter2.prototype.removeAllListeners = function removeAllListeners(event) {
    var evt;
    if (event) {
      evt = prefix ? prefix + event : event;
      if (this._events[evt]) clearEvent(this, evt);
    } else {
      this._events = new Events2();
      this._eventsCount = 0;
    }
    return this;
  };
  EventEmitter2.prototype.off = EventEmitter2.prototype.removeListener;
  EventEmitter2.prototype.addListener = EventEmitter2.prototype.on;
  EventEmitter2.prefixed = prefix;
  EventEmitter2.EventEmitter = EventEmitter2;
  {
    module.exports = EventEmitter2;
  }
})(eventemitter3);
var eventemitter3Exports = eventemitter3.exports;
var EventEmitter = /* @__PURE__ */ getDefaultExportFromCjs(eventemitter3Exports);
class TransmuxerInterface {
  constructor(hls, id, onTransmuxComplete, onFlush) {
    this.error = null;
    this.hls = void 0;
    this.id = void 0;
    this.observer = void 0;
    this.frag = null;
    this.part = null;
    this.useWorker = void 0;
    this.workerContext = null;
    this.onwmsg = void 0;
    this.transmuxer = null;
    this.onTransmuxComplete = void 0;
    this.onFlush = void 0;
    const config = hls.config;
    this.hls = hls;
    this.id = id;
    this.useWorker = !!config.enableWorker;
    this.onTransmuxComplete = onTransmuxComplete;
    this.onFlush = onFlush;
    const forwardMessage = (ev, data) => {
      data = data || {};
      data.frag = this.frag;
      data.id = this.id;
      if (ev === Events.ERROR) {
        this.error = data.error;
      }
      this.hls.trigger(ev, data);
    };
    this.observer = new EventEmitter();
    this.observer.on(Events.FRAG_DECRYPTED, forwardMessage);
    this.observer.on(Events.ERROR, forwardMessage);
    const MediaSource = getMediaSource(config.preferManagedMediaSource) || {
      isTypeSupported: () => false
    };
    const m2tsTypeSupported = {
      mpeg: MediaSource.isTypeSupported("audio/mpeg"),
      mp3: MediaSource.isTypeSupported('audio/mp4; codecs="mp3"'),
      ac3: MediaSource.isTypeSupported('audio/mp4; codecs="ac-3"')
    };
    if (this.useWorker && typeof Worker !== "undefined") {
      const canCreateWorker = config.workerPath || hasUMDWorker();
      if (canCreateWorker) {
        try {
          if (config.workerPath) {
            logger.log(`loading Web Worker ${config.workerPath} for "${id}"`);
            this.workerContext = loadWorker(config.workerPath);
          } else {
            logger.log(`injecting Web Worker for "${id}"`);
            this.workerContext = injectWorker();
          }
          this.onwmsg = (event) => this.onWorkerMessage(event);
          const {
            worker
          } = this.workerContext;
          worker.addEventListener("message", this.onwmsg);
          worker.onerror = (event) => {
            const error = new Error(`${event.message}  (${event.filename}:${event.lineno})`);
            config.enableWorker = false;
            logger.warn(`Error in "${id}" Web Worker, fallback to inline`);
            this.hls.trigger(Events.ERROR, {
              type: ErrorTypes.OTHER_ERROR,
              details: ErrorDetails.INTERNAL_EXCEPTION,
              fatal: false,
              event: "demuxerWorker",
              error
            });
          };
          worker.postMessage({
            cmd: "init",
            typeSupported: m2tsTypeSupported,
            vendor: "",
            id,
            config: JSON.stringify(config)
          });
        } catch (err) {
          logger.warn(`Error setting up "${id}" Web Worker, fallback to inline`, err);
          this.resetWorker();
          this.error = null;
          this.transmuxer = new Transmuxer(this.observer, m2tsTypeSupported, config, "", id);
        }
        return;
      }
    }
    this.transmuxer = new Transmuxer(this.observer, m2tsTypeSupported, config, "", id);
  }
  resetWorker() {
    if (this.workerContext) {
      const {
        worker,
        objectURL
      } = this.workerContext;
      if (objectURL) {
        self.URL.revokeObjectURL(objectURL);
      }
      worker.removeEventListener("message", this.onwmsg);
      worker.onerror = null;
      worker.terminate();
      this.workerContext = null;
    }
  }
  destroy() {
    if (this.workerContext) {
      this.resetWorker();
      this.onwmsg = void 0;
    } else {
      const transmuxer = this.transmuxer;
      if (transmuxer) {
        transmuxer.destroy();
        this.transmuxer = null;
      }
    }
    const observer = this.observer;
    if (observer) {
      observer.removeAllListeners();
    }
    this.frag = null;
    this.observer = null;
    this.hls = null;
  }
  push(data, initSegmentData, audioCodec, videoCodec, frag, part, duration, accurateTimeOffset, chunkMeta, defaultInitPTS) {
    var _frag$initSegment, _lastFrag$initSegment;
    chunkMeta.transmuxing.start = self.performance.now();
    const {
      transmuxer
    } = this;
    const timeOffset = part ? part.start : frag.start;
    const decryptdata = frag.decryptdata;
    const lastFrag = this.frag;
    const discontinuity = !(lastFrag && frag.cc === lastFrag.cc);
    const trackSwitch = !(lastFrag && chunkMeta.level === lastFrag.level);
    const snDiff = lastFrag ? chunkMeta.sn - lastFrag.sn : -1;
    const partDiff = this.part ? chunkMeta.part - this.part.index : -1;
    const progressive = snDiff === 0 && chunkMeta.id > 1 && chunkMeta.id === (lastFrag == null ? void 0 : lastFrag.stats.chunkCount);
    const contiguous = !trackSwitch && (snDiff === 1 || snDiff === 0 && (partDiff === 1 || progressive && partDiff <= 0));
    const now2 = self.performance.now();
    if (trackSwitch || snDiff || frag.stats.parsing.start === 0) {
      frag.stats.parsing.start = now2;
    }
    if (part && (partDiff || !contiguous)) {
      part.stats.parsing.start = now2;
    }
    const initSegmentChange = !(lastFrag && ((_frag$initSegment = frag.initSegment) == null ? void 0 : _frag$initSegment.url) === ((_lastFrag$initSegment = lastFrag.initSegment) == null ? void 0 : _lastFrag$initSegment.url));
    const state = new TransmuxState(discontinuity, contiguous, accurateTimeOffset, trackSwitch, timeOffset, initSegmentChange);
    if (!contiguous || discontinuity || initSegmentChange) {
      logger.log(`[transmuxer-interface, ${frag.type}]: Starting new transmux session for sn: ${chunkMeta.sn} p: ${chunkMeta.part} level: ${chunkMeta.level} id: ${chunkMeta.id}
        discontinuity: ${discontinuity}
        trackSwitch: ${trackSwitch}
        contiguous: ${contiguous}
        accurateTimeOffset: ${accurateTimeOffset}
        timeOffset: ${timeOffset}
        initSegmentChange: ${initSegmentChange}`);
      const config = new TransmuxConfig(audioCodec, videoCodec, initSegmentData, duration, defaultInitPTS);
      this.configureTransmuxer(config);
    }
    this.frag = frag;
    this.part = part;
    if (this.workerContext) {
      this.workerContext.worker.postMessage({
        cmd: "demux",
        data,
        decryptdata,
        chunkMeta,
        state
      }, data instanceof ArrayBuffer ? [data] : []);
    } else if (transmuxer) {
      const transmuxResult = transmuxer.push(data, decryptdata, chunkMeta, state);
      if (isPromise(transmuxResult)) {
        transmuxer.async = true;
        transmuxResult.then((data2) => {
          this.handleTransmuxComplete(data2);
        }).catch((error) => {
          this.transmuxerError(error, chunkMeta, "transmuxer-interface push error");
        });
      } else {
        transmuxer.async = false;
        this.handleTransmuxComplete(transmuxResult);
      }
    }
  }
  flush(chunkMeta) {
    chunkMeta.transmuxing.start = self.performance.now();
    const {
      transmuxer
    } = this;
    if (this.workerContext) {
      this.workerContext.worker.postMessage({
        cmd: "flush",
        chunkMeta
      });
    } else if (transmuxer) {
      let transmuxResult = transmuxer.flush(chunkMeta);
      const asyncFlush = isPromise(transmuxResult);
      if (asyncFlush || transmuxer.async) {
        if (!isPromise(transmuxResult)) {
          transmuxResult = Promise.resolve(transmuxResult);
        }
        transmuxResult.then((data) => {
          this.handleFlushResult(data, chunkMeta);
        }).catch((error) => {
          this.transmuxerError(error, chunkMeta, "transmuxer-interface flush error");
        });
      } else {
        this.handleFlushResult(transmuxResult, chunkMeta);
      }
    }
  }
  transmuxerError(error, chunkMeta, reason) {
    if (!this.hls) {
      return;
    }
    this.error = error;
    this.hls.trigger(Events.ERROR, {
      type: ErrorTypes.MEDIA_ERROR,
      details: ErrorDetails.FRAG_PARSING_ERROR,
      chunkMeta,
      frag: this.frag || void 0,
      fatal: false,
      error,
      err: error,
      reason
    });
  }
  handleFlushResult(results, chunkMeta) {
    results.forEach((result) => {
      this.handleTransmuxComplete(result);
    });
    this.onFlush(chunkMeta);
  }
  onWorkerMessage(event) {
    const data = event.data;
    if (!(data != null && data.event)) {
      logger.warn(`worker message received with no ${data ? "event name" : "data"}`);
      return;
    }
    const hls = this.hls;
    if (!this.hls) {
      return;
    }
    switch (data.event) {
      case "init": {
        var _this$workerContext;
        const objectURL = (_this$workerContext = this.workerContext) == null ? void 0 : _this$workerContext.objectURL;
        if (objectURL) {
          self.URL.revokeObjectURL(objectURL);
        }
        break;
      }
      case "transmuxComplete": {
        this.handleTransmuxComplete(data.data);
        break;
      }
      case "flush": {
        this.onFlush(data.data);
        break;
      }
      // pass logs from the worker thread to the main logger
      case "workerLog":
        if (logger[data.data.logType]) {
          logger[data.data.logType](data.data.message);
        }
        break;
      default: {
        data.data = data.data || {};
        data.data.frag = this.frag;
        data.data.id = this.id;
        hls.trigger(data.event, data.data);
        break;
      }
    }
  }
  configureTransmuxer(config) {
    const {
      transmuxer
    } = this;
    if (this.workerContext) {
      this.workerContext.worker.postMessage({
        cmd: "configure",
        config
      });
    } else if (transmuxer) {
      transmuxer.configure(config);
    }
  }
  handleTransmuxComplete(result) {
    result.chunkMeta.transmuxing.end = self.performance.now();
    this.onTransmuxComplete(result);
  }
}
function subtitleOptionsIdentical(trackList1, trackList2) {
  if (trackList1.length !== trackList2.length) {
    return false;
  }
  for (let i2 = 0; i2 < trackList1.length; i2++) {
    if (!mediaAttributesIdentical(trackList1[i2].attrs, trackList2[i2].attrs)) {
      return false;
    }
  }
  return true;
}
function mediaAttributesIdentical(attrs1, attrs2, customAttributes) {
  const stableRenditionId = attrs1["STABLE-RENDITION-ID"];
  if (stableRenditionId && !customAttributes) {
    return stableRenditionId === attrs2["STABLE-RENDITION-ID"];
  }
  return !(customAttributes || ["LANGUAGE", "NAME", "CHARACTERISTICS", "AUTOSELECT", "DEFAULT", "FORCED", "ASSOC-LANGUAGE"]).some((subtitleAttribute) => attrs1[subtitleAttribute] !== attrs2[subtitleAttribute]);
}
function subtitleTrackMatchesTextTrack(subtitleTrack, textTrack) {
  return textTrack.label.toLowerCase() === subtitleTrack.name.toLowerCase() && (!textTrack.language || textTrack.language.toLowerCase() === (subtitleTrack.lang || "").toLowerCase());
}
const TICK_INTERVAL$2 = 100;
class AudioStreamController extends BaseStreamController {
  constructor(hls, fragmentTracker, keyLoader) {
    super(hls, fragmentTracker, keyLoader, "[audio-stream-controller]", PlaylistLevelType.AUDIO);
    this.videoBuffer = null;
    this.videoTrackCC = -1;
    this.waitingVideoCC = -1;
    this.bufferedTrack = null;
    this.switchingTrack = null;
    this.trackId = -1;
    this.waitingData = null;
    this.mainDetails = null;
    this.flushing = false;
    this.bufferFlushed = false;
    this.cachedTrackLoadedData = null;
    this._registerListeners();
  }
  onHandlerDestroying() {
    this._unregisterListeners();
    super.onHandlerDestroying();
    this.mainDetails = null;
    this.bufferedTrack = null;
    this.switchingTrack = null;
  }
  _registerListeners() {
    const {
      hls
    } = this;
    hls.on(Events.MEDIA_ATTACHED, this.onMediaAttached, this);
    hls.on(Events.MEDIA_DETACHING, this.onMediaDetaching, this);
    hls.on(Events.MANIFEST_LOADING, this.onManifestLoading, this);
    hls.on(Events.LEVEL_LOADED, this.onLevelLoaded, this);
    hls.on(Events.AUDIO_TRACKS_UPDATED, this.onAudioTracksUpdated, this);
    hls.on(Events.AUDIO_TRACK_SWITCHING, this.onAudioTrackSwitching, this);
    hls.on(Events.AUDIO_TRACK_LOADED, this.onAudioTrackLoaded, this);
    hls.on(Events.ERROR, this.onError, this);
    hls.on(Events.BUFFER_RESET, this.onBufferReset, this);
    hls.on(Events.BUFFER_CREATED, this.onBufferCreated, this);
    hls.on(Events.BUFFER_FLUSHING, this.onBufferFlushing, this);
    hls.on(Events.BUFFER_FLUSHED, this.onBufferFlushed, this);
    hls.on(Events.INIT_PTS_FOUND, this.onInitPtsFound, this);
    hls.on(Events.FRAG_BUFFERED, this.onFragBuffered, this);
  }
  _unregisterListeners() {
    const {
      hls
    } = this;
    hls.off(Events.MEDIA_ATTACHED, this.onMediaAttached, this);
    hls.off(Events.MEDIA_DETACHING, this.onMediaDetaching, this);
    hls.off(Events.MANIFEST_LOADING, this.onManifestLoading, this);
    hls.off(Events.LEVEL_LOADED, this.onLevelLoaded, this);
    hls.off(Events.AUDIO_TRACKS_UPDATED, this.onAudioTracksUpdated, this);
    hls.off(Events.AUDIO_TRACK_SWITCHING, this.onAudioTrackSwitching, this);
    hls.off(Events.AUDIO_TRACK_LOADED, this.onAudioTrackLoaded, this);
    hls.off(Events.ERROR, this.onError, this);
    hls.off(Events.BUFFER_RESET, this.onBufferReset, this);
    hls.off(Events.BUFFER_CREATED, this.onBufferCreated, this);
    hls.off(Events.BUFFER_FLUSHING, this.onBufferFlushing, this);
    hls.off(Events.BUFFER_FLUSHED, this.onBufferFlushed, this);
    hls.off(Events.INIT_PTS_FOUND, this.onInitPtsFound, this);
    hls.off(Events.FRAG_BUFFERED, this.onFragBuffered, this);
  }
  // INIT_PTS_FOUND is triggered when the video track parsed in the stream-controller has a new PTS value
  onInitPtsFound(event, {
    frag,
    id,
    initPTS,
    timescale
  }) {
    if (id === "main") {
      const cc = frag.cc;
      this.initPTS[frag.cc] = {
        baseTime: initPTS,
        timescale
      };
      this.log(`InitPTS for cc: ${cc} found from main: ${initPTS}`);
      this.videoTrackCC = cc;
      if (this.state === State.WAITING_INIT_PTS) {
        this.tick();
      }
    }
  }
  startLoad(startPosition) {
    if (!this.levels) {
      this.startPosition = startPosition;
      this.state = State.STOPPED;
      return;
    }
    const lastCurrentTime = this.lastCurrentTime;
    this.stopLoad();
    this.setInterval(TICK_INTERVAL$2);
    if (lastCurrentTime > 0 && startPosition === -1) {
      this.log(`Override startPosition with lastCurrentTime @${lastCurrentTime.toFixed(3)}`);
      startPosition = lastCurrentTime;
      this.state = State.IDLE;
    } else {
      this.loadedmetadata = false;
      this.state = State.WAITING_TRACK;
    }
    this.nextLoadPosition = this.startPosition = this.lastCurrentTime = startPosition;
    this.tick();
  }
  doTick() {
    switch (this.state) {
      case State.IDLE:
        this.doTickIdle();
        break;
      case State.WAITING_TRACK: {
        var _levels$trackId;
        const {
          levels,
          trackId
        } = this;
        const details = levels == null ? void 0 : (_levels$trackId = levels[trackId]) == null ? void 0 : _levels$trackId.details;
        if (details) {
          if (this.waitForCdnTuneIn(details)) {
            break;
          }
          this.state = State.WAITING_INIT_PTS;
        }
        break;
      }
      case State.FRAG_LOADING_WAITING_RETRY: {
        var _this$media;
        const now2 = performance.now();
        const retryDate = this.retryDate;
        if (!retryDate || now2 >= retryDate || (_this$media = this.media) != null && _this$media.seeking) {
          const {
            levels,
            trackId
          } = this;
          this.log("RetryDate reached, switch back to IDLE state");
          this.resetStartWhenNotLoaded((levels == null ? void 0 : levels[trackId]) || null);
          this.state = State.IDLE;
        }
        break;
      }
      case State.WAITING_INIT_PTS: {
        const waitingData = this.waitingData;
        if (waitingData) {
          const {
            frag,
            part,
            cache,
            complete
          } = waitingData;
          if (this.initPTS[frag.cc] !== void 0) {
            this.waitingData = null;
            this.waitingVideoCC = -1;
            this.state = State.FRAG_LOADING;
            const payload = cache.flush();
            const data = {
              frag,
              part,
              payload,
              networkDetails: null
            };
            this._handleFragmentLoadProgress(data);
            if (complete) {
              super._handleFragmentLoadComplete(data);
            }
          } else if (this.videoTrackCC !== this.waitingVideoCC) {
            this.log(`Waiting fragment cc (${frag.cc}) cancelled because video is at cc ${this.videoTrackCC}`);
            this.clearWaitingFragment();
          } else {
            const pos = this.getLoadPosition();
            const bufferInfo = BufferHelper.bufferInfo(this.mediaBuffer, pos, this.config.maxBufferHole);
            const waitingFragmentAtPosition = fragmentWithinToleranceTest(bufferInfo.end, this.config.maxFragLookUpTolerance, frag);
            if (waitingFragmentAtPosition < 0) {
              this.log(`Waiting fragment cc (${frag.cc}) @ ${frag.start} cancelled because another fragment at ${bufferInfo.end} is needed`);
              this.clearWaitingFragment();
            }
          }
        } else {
          this.state = State.IDLE;
        }
      }
    }
    this.onTickEnd();
  }
  clearWaitingFragment() {
    const waitingData = this.waitingData;
    if (waitingData) {
      this.fragmentTracker.removeFragment(waitingData.frag);
      this.waitingData = null;
      this.waitingVideoCC = -1;
      this.state = State.IDLE;
    }
  }
  resetLoadingState() {
    this.clearWaitingFragment();
    super.resetLoadingState();
  }
  onTickEnd() {
    const {
      media
    } = this;
    if (!(media != null && media.readyState)) {
      return;
    }
    this.lastCurrentTime = media.currentTime;
  }
  doTickIdle() {
    const {
      hls,
      levels,
      media,
      trackId
    } = this;
    const config = hls.config;
    if (!media && (this.startFragRequested || !config.startFragPrefetch) || !(levels != null && levels[trackId])) {
      return;
    }
    const levelInfo = levels[trackId];
    const trackDetails = levelInfo.details;
    if (!trackDetails || trackDetails.live && this.levelLastLoaded !== levelInfo || this.waitForCdnTuneIn(trackDetails)) {
      this.state = State.WAITING_TRACK;
      return;
    }
    const bufferable = this.mediaBuffer ? this.mediaBuffer : this.media;
    if (this.bufferFlushed && bufferable) {
      this.bufferFlushed = false;
      this.afterBufferFlushed(bufferable, ElementaryStreamTypes.AUDIO, PlaylistLevelType.AUDIO);
    }
    const bufferInfo = this.getFwdBufferInfo(bufferable, PlaylistLevelType.AUDIO);
    if (bufferInfo === null) {
      return;
    }
    const {
      bufferedTrack,
      switchingTrack
    } = this;
    if (!switchingTrack && this._streamEnded(bufferInfo, trackDetails)) {
      hls.trigger(Events.BUFFER_EOS, {
        type: "audio"
      });
      this.state = State.ENDED;
      return;
    }
    const mainBufferInfo = this.getFwdBufferInfo(this.videoBuffer ? this.videoBuffer : this.media, PlaylistLevelType.MAIN);
    const bufferLen = bufferInfo.len;
    const maxBufLen = this.getMaxBufferLength(mainBufferInfo == null ? void 0 : mainBufferInfo.len);
    const fragments = trackDetails.fragments;
    const start = fragments[0].start;
    let targetBufferTime = this.flushing ? this.getLoadPosition() : bufferInfo.end;
    if (switchingTrack && media) {
      const pos = this.getLoadPosition();
      if (bufferedTrack && !mediaAttributesIdentical(switchingTrack.attrs, bufferedTrack.attrs)) {
        targetBufferTime = pos;
      }
      if (trackDetails.PTSKnown && pos < start) {
        if (bufferInfo.end > start || bufferInfo.nextStart) {
          this.log("Alt audio track ahead of main track, seek to start of alt audio track");
          media.currentTime = start + 0.05;
        }
      }
    }
    if (bufferLen >= maxBufLen && !switchingTrack && targetBufferTime < fragments[fragments.length - 1].start) {
      return;
    }
    let frag = this.getNextFragment(targetBufferTime, trackDetails);
    let atGap = false;
    if (frag && this.isLoopLoading(frag, targetBufferTime)) {
      atGap = !!frag.gap;
      frag = this.getNextFragmentLoopLoading(frag, trackDetails, bufferInfo, PlaylistLevelType.MAIN, maxBufLen);
    }
    if (!frag) {
      this.bufferFlushed = true;
      return;
    }
    const atBufferSyncLimit = mainBufferInfo && frag.start > mainBufferInfo.end + trackDetails.targetduration;
    if (atBufferSyncLimit || // Or wait for main buffer after buffing some audio
    !(mainBufferInfo != null && mainBufferInfo.len) && bufferInfo.len) {
      const mainFrag = this.getAppendedFrag(frag.start, PlaylistLevelType.MAIN);
      if (mainFrag === null) {
        return;
      }
      atGap || (atGap = !!mainFrag.gap || !!atBufferSyncLimit && mainBufferInfo.len === 0);
      if (atBufferSyncLimit && !atGap || atGap && bufferInfo.nextStart && bufferInfo.nextStart < mainFrag.end) {
        return;
      }
    }
    this.loadFragment(frag, levelInfo, targetBufferTime);
  }
  getMaxBufferLength(mainBufferLength) {
    const maxConfigBuffer = super.getMaxBufferLength();
    if (!mainBufferLength) {
      return maxConfigBuffer;
    }
    return Math.min(Math.max(maxConfigBuffer, mainBufferLength), this.config.maxMaxBufferLength);
  }
  onMediaDetaching() {
    this.videoBuffer = null;
    this.bufferFlushed = this.flushing = false;
    super.onMediaDetaching();
  }
  onAudioTracksUpdated(event, {
    audioTracks
  }) {
    this.resetTransmuxer();
    this.levels = audioTracks.map((mediaPlaylist) => new Level(mediaPlaylist));
  }
  onAudioTrackSwitching(event, data) {
    const altAudio = !!data.url;
    this.trackId = data.id;
    const {
      fragCurrent
    } = this;
    if (fragCurrent) {
      fragCurrent.abortRequests();
      this.removeUnbufferedFrags(fragCurrent.start);
    }
    this.resetLoadingState();
    if (!altAudio) {
      this.resetTransmuxer();
    } else {
      this.setInterval(TICK_INTERVAL$2);
    }
    if (altAudio) {
      this.switchingTrack = data;
      this.state = State.IDLE;
      this.flushAudioIfNeeded(data);
    } else {
      this.switchingTrack = null;
      this.bufferedTrack = data;
      this.state = State.STOPPED;
    }
    this.tick();
  }
  onManifestLoading() {
    this.fragmentTracker.removeAllFragments();
    this.startPosition = this.lastCurrentTime = 0;
    this.bufferFlushed = this.flushing = false;
    this.levels = this.mainDetails = this.waitingData = this.bufferedTrack = this.cachedTrackLoadedData = this.switchingTrack = null;
    this.startFragRequested = false;
    this.trackId = this.videoTrackCC = this.waitingVideoCC = -1;
  }
  onLevelLoaded(event, data) {
    this.mainDetails = data.details;
    if (this.cachedTrackLoadedData !== null) {
      this.hls.trigger(Events.AUDIO_TRACK_LOADED, this.cachedTrackLoadedData);
      this.cachedTrackLoadedData = null;
    }
  }
  onAudioTrackLoaded(event, data) {
    var _track$details;
    if (this.mainDetails == null) {
      this.cachedTrackLoadedData = data;
      return;
    }
    const {
      levels
    } = this;
    const {
      details: newDetails,
      id: trackId
    } = data;
    if (!levels) {
      this.warn(`Audio tracks were reset while loading level ${trackId}`);
      return;
    }
    this.log(`Audio track ${trackId} loaded [${newDetails.startSN},${newDetails.endSN}]${newDetails.lastPartSn ? `[part-${newDetails.lastPartSn}-${newDetails.lastPartIndex}]` : ""},duration:${newDetails.totalduration}`);
    const track = levels[trackId];
    let sliding = 0;
    if (newDetails.live || (_track$details = track.details) != null && _track$details.live) {
      this.checkLiveUpdate(newDetails);
      const mainDetails = this.mainDetails;
      if (newDetails.deltaUpdateFailed || !mainDetails) {
        return;
      }
      if (!track.details && newDetails.hasProgramDateTime && mainDetails.hasProgramDateTime) {
        alignMediaPlaylistByPDT(newDetails, mainDetails);
        sliding = newDetails.fragments[0].start;
      } else {
        var _this$levelLastLoaded;
        sliding = this.alignPlaylists(newDetails, track.details, (_this$levelLastLoaded = this.levelLastLoaded) == null ? void 0 : _this$levelLastLoaded.details);
      }
    }
    track.details = newDetails;
    this.levelLastLoaded = track;
    if (!this.startFragRequested && (this.mainDetails || !newDetails.live)) {
      this.setStartPosition(this.mainDetails || newDetails, sliding);
    }
    if (this.state === State.WAITING_TRACK && !this.waitForCdnTuneIn(newDetails)) {
      this.state = State.IDLE;
    }
    this.tick();
  }
  _handleFragmentLoadProgress(data) {
    var _frag$initSegment;
    const {
      frag,
      part,
      payload
    } = data;
    const {
      config,
      trackId,
      levels
    } = this;
    if (!levels) {
      this.warn(`Audio tracks were reset while fragment load was in progress. Fragment ${frag.sn} of level ${frag.level} will not be buffered`);
      return;
    }
    const track = levels[trackId];
    if (!track) {
      this.warn("Audio track is undefined on fragment load progress");
      return;
    }
    const details = track.details;
    if (!details) {
      this.warn("Audio track details undefined on fragment load progress");
      this.removeUnbufferedFrags(frag.start);
      return;
    }
    const audioCodec = config.defaultAudioCodec || track.audioCodec || "mp4a.40.2";
    let transmuxer = this.transmuxer;
    if (!transmuxer) {
      transmuxer = this.transmuxer = new TransmuxerInterface(this.hls, PlaylistLevelType.AUDIO, this._handleTransmuxComplete.bind(this), this._handleTransmuxerFlush.bind(this));
    }
    const initPTS = this.initPTS[frag.cc];
    const initSegmentData = (_frag$initSegment = frag.initSegment) == null ? void 0 : _frag$initSegment.data;
    if (initPTS !== void 0) {
      const accurateTimeOffset = false;
      const partIndex = part ? part.index : -1;
      const partial = partIndex !== -1;
      const chunkMeta = new ChunkMetadata(frag.level, frag.sn, frag.stats.chunkCount, payload.byteLength, partIndex, partial);
      transmuxer.push(payload, initSegmentData, audioCodec, "", frag, part, details.totalduration, accurateTimeOffset, chunkMeta, initPTS);
    } else {
      this.log(`Unknown video PTS for cc ${frag.cc}, waiting for video PTS before demuxing audio frag ${frag.sn} of [${details.startSN} ,${details.endSN}],track ${trackId}`);
      const {
        cache
      } = this.waitingData = this.waitingData || {
        frag,
        part,
        cache: new ChunkCache(),
        complete: false
      };
      cache.push(new Uint8Array(payload));
      this.waitingVideoCC = this.videoTrackCC;
      this.state = State.WAITING_INIT_PTS;
    }
  }
  _handleFragmentLoadComplete(fragLoadedData) {
    if (this.waitingData) {
      this.waitingData.complete = true;
      return;
    }
    super._handleFragmentLoadComplete(fragLoadedData);
  }
  onBufferReset() {
    this.mediaBuffer = this.videoBuffer = null;
    this.loadedmetadata = false;
  }
  onBufferCreated(event, data) {
    const audioTrack = data.tracks.audio;
    if (audioTrack) {
      this.mediaBuffer = audioTrack.buffer || null;
    }
    if (data.tracks.video) {
      this.videoBuffer = data.tracks.video.buffer || null;
    }
  }
  onFragBuffered(event, data) {
    const {
      frag,
      part
    } = data;
    if (frag.type !== PlaylistLevelType.AUDIO) {
      if (!this.loadedmetadata && frag.type === PlaylistLevelType.MAIN) {
        const bufferable = this.videoBuffer || this.media;
        if (bufferable) {
          const bufferedTimeRanges = BufferHelper.getBuffered(bufferable);
          if (bufferedTimeRanges.length) {
            this.loadedmetadata = true;
          }
        }
      }
      return;
    }
    if (this.fragContextChanged(frag)) {
      this.warn(`Fragment ${frag.sn}${part ? " p: " + part.index : ""} of level ${frag.level} finished buffering, but was aborted. state: ${this.state}, audioSwitch: ${this.switchingTrack ? this.switchingTrack.name : "false"}`);
      return;
    }
    if (frag.sn !== "initSegment") {
      this.fragPrevious = frag;
      const track = this.switchingTrack;
      if (track) {
        this.bufferedTrack = track;
        this.switchingTrack = null;
        this.hls.trigger(Events.AUDIO_TRACK_SWITCHED, _objectSpread2({}, track));
      }
    }
    this.fragBufferedComplete(frag, part);
  }
  onError(event, data) {
    var _data$context;
    if (data.fatal) {
      this.state = State.ERROR;
      return;
    }
    switch (data.details) {
      case ErrorDetails.FRAG_GAP:
      case ErrorDetails.FRAG_PARSING_ERROR:
      case ErrorDetails.FRAG_DECRYPT_ERROR:
      case ErrorDetails.FRAG_LOAD_ERROR:
      case ErrorDetails.FRAG_LOAD_TIMEOUT:
      case ErrorDetails.KEY_LOAD_ERROR:
      case ErrorDetails.KEY_LOAD_TIMEOUT:
        this.onFragmentOrKeyLoadError(PlaylistLevelType.AUDIO, data);
        break;
      case ErrorDetails.AUDIO_TRACK_LOAD_ERROR:
      case ErrorDetails.AUDIO_TRACK_LOAD_TIMEOUT:
      case ErrorDetails.LEVEL_PARSING_ERROR:
        if (!data.levelRetry && this.state === State.WAITING_TRACK && ((_data$context = data.context) == null ? void 0 : _data$context.type) === PlaylistContextType.AUDIO_TRACK) {
          this.state = State.IDLE;
        }
        break;
      case ErrorDetails.BUFFER_APPEND_ERROR:
      case ErrorDetails.BUFFER_FULL_ERROR:
        if (!data.parent || data.parent !== "audio") {
          return;
        }
        if (data.details === ErrorDetails.BUFFER_APPEND_ERROR) {
          this.resetLoadingState();
          return;
        }
        if (this.reduceLengthAndFlushBuffer(data)) {
          this.bufferedTrack = null;
          super.flushMainBuffer(0, Number.POSITIVE_INFINITY, "audio");
        }
        break;
      case ErrorDetails.INTERNAL_EXCEPTION:
        this.recoverWorkerError(data);
        break;
    }
  }
  onBufferFlushing(event, {
    type
  }) {
    if (type !== ElementaryStreamTypes.VIDEO) {
      this.flushing = true;
    }
  }
  onBufferFlushed(event, {
    type
  }) {
    if (type !== ElementaryStreamTypes.VIDEO) {
      this.flushing = false;
      this.bufferFlushed = true;
      if (this.state === State.ENDED) {
        this.state = State.IDLE;
      }
      const mediaBuffer = this.mediaBuffer || this.media;
      if (mediaBuffer) {
        this.afterBufferFlushed(mediaBuffer, type, PlaylistLevelType.AUDIO);
        this.tick();
      }
    }
  }
  _handleTransmuxComplete(transmuxResult) {
    var _id3$samples;
    const id = "audio";
    const {
      hls
    } = this;
    const {
      remuxResult,
      chunkMeta
    } = transmuxResult;
    const context = this.getCurrentContext(chunkMeta);
    if (!context) {
      this.resetWhenMissingContext(chunkMeta);
      return;
    }
    const {
      frag,
      part,
      level
    } = context;
    const {
      details
    } = level;
    const {
      audio,
      text,
      id3,
      initSegment
    } = remuxResult;
    if (this.fragContextChanged(frag) || !details) {
      this.fragmentTracker.removeFragment(frag);
      return;
    }
    this.state = State.PARSING;
    if (this.switchingTrack && audio) {
      this.completeAudioSwitch(this.switchingTrack);
    }
    if (initSegment != null && initSegment.tracks) {
      const mapFragment = frag.initSegment || frag;
      this._bufferInitSegment(level, initSegment.tracks, mapFragment, chunkMeta);
      hls.trigger(Events.FRAG_PARSING_INIT_SEGMENT, {
        frag: mapFragment,
        id,
        tracks: initSegment.tracks
      });
    }
    if (audio) {
      const {
        startPTS,
        endPTS,
        startDTS,
        endDTS
      } = audio;
      if (part) {
        part.elementaryStreams[ElementaryStreamTypes.AUDIO] = {
          startPTS,
          endPTS,
          startDTS,
          endDTS
        };
      }
      frag.setElementaryStreamInfo(ElementaryStreamTypes.AUDIO, startPTS, endPTS, startDTS, endDTS);
      this.bufferFragmentData(audio, frag, part, chunkMeta);
    }
    if (id3 != null && (_id3$samples = id3.samples) != null && _id3$samples.length) {
      const emittedID3 = _extends({
        id,
        frag,
        details
      }, id3);
      hls.trigger(Events.FRAG_PARSING_METADATA, emittedID3);
    }
    if (text) {
      const emittedText = _extends({
        id,
        frag,
        details
      }, text);
      hls.trigger(Events.FRAG_PARSING_USERDATA, emittedText);
    }
  }
  _bufferInitSegment(currentLevel, tracks, frag, chunkMeta) {
    if (this.state !== State.PARSING) {
      return;
    }
    if (tracks.video) {
      delete tracks.video;
    }
    const track = tracks.audio;
    if (!track) {
      return;
    }
    track.id = "audio";
    const variantAudioCodecs = currentLevel.audioCodec;
    this.log(`Init audio buffer, container:${track.container}, codecs[level/parsed]=[${variantAudioCodecs}/${track.codec}]`);
    if (variantAudioCodecs && variantAudioCodecs.split(",").length === 1) {
      track.levelCodec = variantAudioCodecs;
    }
    this.hls.trigger(Events.BUFFER_CODECS, tracks);
    const initSegment = track.initSegment;
    if (initSegment != null && initSegment.byteLength) {
      const segment = {
        type: "audio",
        frag,
        part: null,
        chunkMeta,
        parent: frag.type,
        data: initSegment
      };
      this.hls.trigger(Events.BUFFER_APPENDING, segment);
    }
    this.tickImmediate();
  }
  loadFragment(frag, track, targetBufferTime) {
    const fragState = this.fragmentTracker.getState(frag);
    this.fragCurrent = frag;
    if (this.switchingTrack || fragState === FragmentState.NOT_LOADED || fragState === FragmentState.PARTIAL) {
      var _track$details2;
      if (frag.sn === "initSegment") {
        this._loadInitSegment(frag, track);
      } else if ((_track$details2 = track.details) != null && _track$details2.live && !this.initPTS[frag.cc]) {
        this.log(`Waiting for video PTS in continuity counter ${frag.cc} of live stream before loading audio fragment ${frag.sn} of level ${this.trackId}`);
        this.state = State.WAITING_INIT_PTS;
        const mainDetails = this.mainDetails;
        if (mainDetails && mainDetails.fragments[0].start !== track.details.fragments[0].start) {
          alignMediaPlaylistByPDT(track.details, mainDetails);
        }
      } else {
        this.startFragRequested = true;
        super.loadFragment(frag, track, targetBufferTime);
      }
    } else {
      this.clearTrackerIfNeeded(frag);
    }
  }
  flushAudioIfNeeded(switchingTrack) {
    const {
      media,
      bufferedTrack
    } = this;
    const bufferedAttributes = bufferedTrack == null ? void 0 : bufferedTrack.attrs;
    const switchAttributes = switchingTrack.attrs;
    if (media && bufferedAttributes && (bufferedAttributes.CHANNELS !== switchAttributes.CHANNELS || bufferedTrack.name !== switchingTrack.name || bufferedTrack.lang !== switchingTrack.lang)) {
      this.log("Switching audio track : flushing all audio");
      super.flushMainBuffer(0, Number.POSITIVE_INFINITY, "audio");
      this.bufferedTrack = null;
    }
  }
  completeAudioSwitch(switchingTrack) {
    const {
      hls
    } = this;
    this.flushAudioIfNeeded(switchingTrack);
    this.bufferedTrack = switchingTrack;
    this.switchingTrack = null;
    hls.trigger(Events.AUDIO_TRACK_SWITCHED, _objectSpread2({}, switchingTrack));
  }
}
class AudioTrackController extends BasePlaylistController {
  constructor(hls) {
    super(hls, "[audio-track-controller]");
    this.tracks = [];
    this.groupIds = null;
    this.tracksInGroup = [];
    this.trackId = -1;
    this.currentTrack = null;
    this.selectDefaultTrack = true;
    this.registerListeners();
  }
  registerListeners() {
    const {
      hls
    } = this;
    hls.on(Events.MANIFEST_LOADING, this.onManifestLoading, this);
    hls.on(Events.MANIFEST_PARSED, this.onManifestParsed, this);
    hls.on(Events.LEVEL_LOADING, this.onLevelLoading, this);
    hls.on(Events.LEVEL_SWITCHING, this.onLevelSwitching, this);
    hls.on(Events.AUDIO_TRACK_LOADED, this.onAudioTrackLoaded, this);
    hls.on(Events.ERROR, this.onError, this);
  }
  unregisterListeners() {
    const {
      hls
    } = this;
    hls.off(Events.MANIFEST_LOADING, this.onManifestLoading, this);
    hls.off(Events.MANIFEST_PARSED, this.onManifestParsed, this);
    hls.off(Events.LEVEL_LOADING, this.onLevelLoading, this);
    hls.off(Events.LEVEL_SWITCHING, this.onLevelSwitching, this);
    hls.off(Events.AUDIO_TRACK_LOADED, this.onAudioTrackLoaded, this);
    hls.off(Events.ERROR, this.onError, this);
  }
  destroy() {
    this.unregisterListeners();
    this.tracks.length = 0;
    this.tracksInGroup.length = 0;
    this.currentTrack = null;
    super.destroy();
  }
  onManifestLoading() {
    this.tracks = [];
    this.tracksInGroup = [];
    this.groupIds = null;
    this.currentTrack = null;
    this.trackId = -1;
    this.selectDefaultTrack = true;
  }
  onManifestParsed(event, data) {
    this.tracks = data.audioTracks || [];
  }
  onAudioTrackLoaded(event, data) {
    const {
      id,
      groupId,
      details
    } = data;
    const trackInActiveGroup = this.tracksInGroup[id];
    if (!trackInActiveGroup || trackInActiveGroup.groupId !== groupId) {
      this.warn(`Audio track with id:${id} and group:${groupId} not found in active group ${trackInActiveGroup == null ? void 0 : trackInActiveGroup.groupId}`);
      return;
    }
    const curDetails = trackInActiveGroup.details;
    trackInActiveGroup.details = data.details;
    this.log(`Audio track ${id} "${trackInActiveGroup.name}" lang:${trackInActiveGroup.lang} group:${groupId} loaded [${details.startSN}-${details.endSN}]`);
    if (id === this.trackId) {
      this.playlistLoaded(id, data, curDetails);
    }
  }
  onLevelLoading(event, data) {
    this.switchLevel(data.level);
  }
  onLevelSwitching(event, data) {
    this.switchLevel(data.level);
  }
  switchLevel(levelIndex) {
    const levelInfo = this.hls.levels[levelIndex];
    if (!levelInfo) {
      return;
    }
    const audioGroups = levelInfo.audioGroups || null;
    const currentGroups = this.groupIds;
    let currentTrack = this.currentTrack;
    if (!audioGroups || (currentGroups == null ? void 0 : currentGroups.length) !== (audioGroups == null ? void 0 : audioGroups.length) || audioGroups != null && audioGroups.some((groupId) => (currentGroups == null ? void 0 : currentGroups.indexOf(groupId)) === -1)) {
      this.groupIds = audioGroups;
      this.trackId = -1;
      this.currentTrack = null;
      const audioTracks = this.tracks.filter((track) => !audioGroups || audioGroups.indexOf(track.groupId) !== -1);
      if (audioTracks.length) {
        if (this.selectDefaultTrack && !audioTracks.some((track) => track.default)) {
          this.selectDefaultTrack = false;
        }
        audioTracks.forEach((track, i2) => {
          track.id = i2;
        });
      } else if (!currentTrack && !this.tracksInGroup.length) {
        return;
      }
      this.tracksInGroup = audioTracks;
      const audioPreference = this.hls.config.audioPreference;
      if (!currentTrack && audioPreference) {
        const groupIndex = findMatchingOption(audioPreference, audioTracks, audioMatchPredicate);
        if (groupIndex > -1) {
          currentTrack = audioTracks[groupIndex];
        } else {
          const allIndex = findMatchingOption(audioPreference, this.tracks);
          currentTrack = this.tracks[allIndex];
        }
      }
      let trackId = this.findTrackId(currentTrack);
      if (trackId === -1 && currentTrack) {
        trackId = this.findTrackId(null);
      }
      const audioTracksUpdated = {
        audioTracks
      };
      this.log(`Updating audio tracks, ${audioTracks.length} track(s) found in group(s): ${audioGroups == null ? void 0 : audioGroups.join(",")}`);
      this.hls.trigger(Events.AUDIO_TRACKS_UPDATED, audioTracksUpdated);
      const selectedTrackId = this.trackId;
      if (trackId !== -1 && selectedTrackId === -1) {
        this.setAudioTrack(trackId);
      } else if (audioTracks.length && selectedTrackId === -1) {
        var _this$groupIds;
        const error = new Error(`No audio track selected for current audio group-ID(s): ${(_this$groupIds = this.groupIds) == null ? void 0 : _this$groupIds.join(",")} track count: ${audioTracks.length}`);
        this.warn(error.message);
        this.hls.trigger(Events.ERROR, {
          type: ErrorTypes.MEDIA_ERROR,
          details: ErrorDetails.AUDIO_TRACK_LOAD_ERROR,
          fatal: true,
          error
        });
      }
    } else if (this.shouldReloadPlaylist(currentTrack)) {
      this.setAudioTrack(this.trackId);
    }
  }
  onError(event, data) {
    if (data.fatal || !data.context) {
      return;
    }
    if (data.context.type === PlaylistContextType.AUDIO_TRACK && data.context.id === this.trackId && (!this.groupIds || this.groupIds.indexOf(data.context.groupId) !== -1)) {
      this.requestScheduled = -1;
      this.checkRetry(data);
    }
  }
  get allAudioTracks() {
    return this.tracks;
  }
  get audioTracks() {
    return this.tracksInGroup;
  }
  get audioTrack() {
    return this.trackId;
  }
  set audioTrack(newId) {
    this.selectDefaultTrack = false;
    this.setAudioTrack(newId);
  }
  setAudioOption(audioOption) {
    const hls = this.hls;
    hls.config.audioPreference = audioOption;
    if (audioOption) {
      const allAudioTracks = this.allAudioTracks;
      this.selectDefaultTrack = false;
      if (allAudioTracks.length) {
        const currentTrack = this.currentTrack;
        if (currentTrack && matchesOption(audioOption, currentTrack, audioMatchPredicate)) {
          return currentTrack;
        }
        const groupIndex = findMatchingOption(audioOption, this.tracksInGroup, audioMatchPredicate);
        if (groupIndex > -1) {
          const track = this.tracksInGroup[groupIndex];
          this.setAudioTrack(groupIndex);
          return track;
        } else if (currentTrack) {
          let searchIndex = hls.loadLevel;
          if (searchIndex === -1) {
            searchIndex = hls.firstAutoLevel;
          }
          const switchIndex = findClosestLevelWithAudioGroup(audioOption, hls.levels, allAudioTracks, searchIndex, audioMatchPredicate);
          if (switchIndex === -1) {
            return null;
          }
          hls.nextLoadLevel = switchIndex;
        }
        if (audioOption.channels || audioOption.audioCodec) {
          const withoutCodecAndChannelsMatch = findMatchingOption(audioOption, allAudioTracks);
          if (withoutCodecAndChannelsMatch > -1) {
            return allAudioTracks[withoutCodecAndChannelsMatch];
          }
        }
      }
    }
    return null;
  }
  setAudioTrack(newId) {
    const tracks = this.tracksInGroup;
    if (newId < 0 || newId >= tracks.length) {
      this.warn(`Invalid audio track id: ${newId}`);
      return;
    }
    this.clearTimer();
    this.selectDefaultTrack = false;
    const lastTrack = this.currentTrack;
    const track = tracks[newId];
    const trackLoaded = track.details && !track.details.live;
    if (newId === this.trackId && track === lastTrack && trackLoaded) {
      return;
    }
    this.log(`Switching to audio-track ${newId} "${track.name}" lang:${track.lang} group:${track.groupId} channels:${track.channels}`);
    this.trackId = newId;
    this.currentTrack = track;
    this.hls.trigger(Events.AUDIO_TRACK_SWITCHING, _objectSpread2({}, track));
    if (trackLoaded) {
      return;
    }
    const hlsUrlParameters = this.switchParams(track.url, lastTrack == null ? void 0 : lastTrack.details, track.details);
    this.loadPlaylist(hlsUrlParameters);
  }
  findTrackId(currentTrack) {
    const audioTracks = this.tracksInGroup;
    for (let i2 = 0; i2 < audioTracks.length; i2++) {
      const track = audioTracks[i2];
      if (this.selectDefaultTrack && !track.default) {
        continue;
      }
      if (!currentTrack || matchesOption(currentTrack, track, audioMatchPredicate)) {
        return i2;
      }
    }
    if (currentTrack) {
      const {
        name,
        lang,
        assocLang,
        characteristics,
        audioCodec,
        channels
      } = currentTrack;
      for (let i2 = 0; i2 < audioTracks.length; i2++) {
        const track = audioTracks[i2];
        if (matchesOption({
          name,
          lang,
          assocLang,
          characteristics,
          audioCodec,
          channels
        }, track, audioMatchPredicate)) {
          return i2;
        }
      }
      for (let i2 = 0; i2 < audioTracks.length; i2++) {
        const track = audioTracks[i2];
        if (mediaAttributesIdentical(currentTrack.attrs, track.attrs, ["LANGUAGE", "ASSOC-LANGUAGE", "CHARACTERISTICS"])) {
          return i2;
        }
      }
      for (let i2 = 0; i2 < audioTracks.length; i2++) {
        const track = audioTracks[i2];
        if (mediaAttributesIdentical(currentTrack.attrs, track.attrs, ["LANGUAGE"])) {
          return i2;
        }
      }
    }
    return -1;
  }
  loadPlaylist(hlsUrlParameters) {
    const audioTrack = this.currentTrack;
    if (this.shouldLoadPlaylist(audioTrack) && audioTrack) {
      super.loadPlaylist();
      const id = audioTrack.id;
      const groupId = audioTrack.groupId;
      let url = audioTrack.url;
      if (hlsUrlParameters) {
        try {
          url = hlsUrlParameters.addDirectives(url);
        } catch (error) {
          this.warn(`Could not construct new URL with HLS Delivery Directives: ${error}`);
        }
      }
      this.log(`loading audio-track playlist ${id} "${audioTrack.name}" lang:${audioTrack.lang} group:${groupId}`);
      this.clearTimer();
      this.hls.trigger(Events.AUDIO_TRACK_LOADING, {
        url,
        id,
        groupId,
        deliveryDirectives: hlsUrlParameters || null
      });
    }
  }
}
const TICK_INTERVAL$1 = 500;
class SubtitleStreamController extends BaseStreamController {
  constructor(hls, fragmentTracker, keyLoader) {
    super(hls, fragmentTracker, keyLoader, "[subtitle-stream-controller]", PlaylistLevelType.SUBTITLE);
    this.currentTrackId = -1;
    this.tracksBuffered = [];
    this.mainDetails = null;
    this._registerListeners();
  }
  onHandlerDestroying() {
    this._unregisterListeners();
    super.onHandlerDestroying();
    this.mainDetails = null;
  }
  _registerListeners() {
    const {
      hls
    } = this;
    hls.on(Events.MEDIA_ATTACHED, this.onMediaAttached, this);
    hls.on(Events.MEDIA_DETACHING, this.onMediaDetaching, this);
    hls.on(Events.MANIFEST_LOADING, this.onManifestLoading, this);
    hls.on(Events.LEVEL_LOADED, this.onLevelLoaded, this);
    hls.on(Events.ERROR, this.onError, this);
    hls.on(Events.SUBTITLE_TRACKS_UPDATED, this.onSubtitleTracksUpdated, this);
    hls.on(Events.SUBTITLE_TRACK_SWITCH, this.onSubtitleTrackSwitch, this);
    hls.on(Events.SUBTITLE_TRACK_LOADED, this.onSubtitleTrackLoaded, this);
    hls.on(Events.SUBTITLE_FRAG_PROCESSED, this.onSubtitleFragProcessed, this);
    hls.on(Events.BUFFER_FLUSHING, this.onBufferFlushing, this);
    hls.on(Events.FRAG_BUFFERED, this.onFragBuffered, this);
  }
  _unregisterListeners() {
    const {
      hls
    } = this;
    hls.off(Events.MEDIA_ATTACHED, this.onMediaAttached, this);
    hls.off(Events.MEDIA_DETACHING, this.onMediaDetaching, this);
    hls.off(Events.MANIFEST_LOADING, this.onManifestLoading, this);
    hls.off(Events.LEVEL_LOADED, this.onLevelLoaded, this);
    hls.off(Events.ERROR, this.onError, this);
    hls.off(Events.SUBTITLE_TRACKS_UPDATED, this.onSubtitleTracksUpdated, this);
    hls.off(Events.SUBTITLE_TRACK_SWITCH, this.onSubtitleTrackSwitch, this);
    hls.off(Events.SUBTITLE_TRACK_LOADED, this.onSubtitleTrackLoaded, this);
    hls.off(Events.SUBTITLE_FRAG_PROCESSED, this.onSubtitleFragProcessed, this);
    hls.off(Events.BUFFER_FLUSHING, this.onBufferFlushing, this);
    hls.off(Events.FRAG_BUFFERED, this.onFragBuffered, this);
  }
  startLoad(startPosition) {
    this.stopLoad();
    this.state = State.IDLE;
    this.setInterval(TICK_INTERVAL$1);
    this.nextLoadPosition = this.startPosition = this.lastCurrentTime = startPosition;
    this.tick();
  }
  onManifestLoading() {
    this.mainDetails = null;
    this.fragmentTracker.removeAllFragments();
  }
  onMediaDetaching() {
    this.tracksBuffered = [];
    super.onMediaDetaching();
  }
  onLevelLoaded(event, data) {
    this.mainDetails = data.details;
  }
  onSubtitleFragProcessed(event, data) {
    const {
      frag,
      success
    } = data;
    this.fragPrevious = frag;
    this.state = State.IDLE;
    if (!success) {
      return;
    }
    const buffered = this.tracksBuffered[this.currentTrackId];
    if (!buffered) {
      return;
    }
    let timeRange;
    const fragStart = frag.start;
    for (let i2 = 0; i2 < buffered.length; i2++) {
      if (fragStart >= buffered[i2].start && fragStart <= buffered[i2].end) {
        timeRange = buffered[i2];
        break;
      }
    }
    const fragEnd = frag.start + frag.duration;
    if (timeRange) {
      timeRange.end = fragEnd;
    } else {
      timeRange = {
        start: fragStart,
        end: fragEnd
      };
      buffered.push(timeRange);
    }
    this.fragmentTracker.fragBuffered(frag);
    this.fragBufferedComplete(frag, null);
  }
  onBufferFlushing(event, data) {
    const {
      startOffset,
      endOffset
    } = data;
    if (startOffset === 0 && endOffset !== Number.POSITIVE_INFINITY) {
      const endOffsetSubtitles = endOffset - 1;
      if (endOffsetSubtitles <= 0) {
        return;
      }
      data.endOffsetSubtitles = Math.max(0, endOffsetSubtitles);
      this.tracksBuffered.forEach((buffered) => {
        for (let i2 = 0; i2 < buffered.length; ) {
          if (buffered[i2].end <= endOffsetSubtitles) {
            buffered.shift();
            continue;
          } else if (buffered[i2].start < endOffsetSubtitles) {
            buffered[i2].start = endOffsetSubtitles;
          } else {
            break;
          }
          i2++;
        }
      });
      this.fragmentTracker.removeFragmentsInRange(startOffset, endOffsetSubtitles, PlaylistLevelType.SUBTITLE);
    }
  }
  onFragBuffered(event, data) {
    if (!this.loadedmetadata && data.frag.type === PlaylistLevelType.MAIN) {
      var _this$media;
      if ((_this$media = this.media) != null && _this$media.buffered.length) {
        this.loadedmetadata = true;
      }
    }
  }
  // If something goes wrong, proceed to next frag, if we were processing one.
  onError(event, data) {
    const frag = data.frag;
    if ((frag == null ? void 0 : frag.type) === PlaylistLevelType.SUBTITLE) {
      if (data.details === ErrorDetails.FRAG_GAP) {
        this.fragmentTracker.fragBuffered(frag, true);
      }
      if (this.fragCurrent) {
        this.fragCurrent.abortRequests();
      }
      if (this.state !== State.STOPPED) {
        this.state = State.IDLE;
      }
    }
  }
  // Got all new subtitle levels.
  onSubtitleTracksUpdated(event, {
    subtitleTracks
  }) {
    if (this.levels && subtitleOptionsIdentical(this.levels, subtitleTracks)) {
      this.levels = subtitleTracks.map((mediaPlaylist) => new Level(mediaPlaylist));
      return;
    }
    this.tracksBuffered = [];
    this.levels = subtitleTracks.map((mediaPlaylist) => {
      const level = new Level(mediaPlaylist);
      this.tracksBuffered[level.id] = [];
      return level;
    });
    this.fragmentTracker.removeFragmentsInRange(0, Number.POSITIVE_INFINITY, PlaylistLevelType.SUBTITLE);
    this.fragPrevious = null;
    this.mediaBuffer = null;
  }
  onSubtitleTrackSwitch(event, data) {
    var _this$levels;
    this.currentTrackId = data.id;
    if (!((_this$levels = this.levels) != null && _this$levels.length) || this.currentTrackId === -1) {
      this.clearInterval();
      return;
    }
    const currentTrack = this.levels[this.currentTrackId];
    if (currentTrack != null && currentTrack.details) {
      this.mediaBuffer = this.mediaBufferTimeRanges;
    } else {
      this.mediaBuffer = null;
    }
    if (currentTrack) {
      this.setInterval(TICK_INTERVAL$1);
    }
  }
  // Got a new set of subtitle fragments.
  onSubtitleTrackLoaded(event, data) {
    var _track$details;
    const {
      currentTrackId,
      levels
    } = this;
    const {
      details: newDetails,
      id: trackId
    } = data;
    if (!levels) {
      this.warn(`Subtitle tracks were reset while loading level ${trackId}`);
      return;
    }
    const track = levels[trackId];
    if (trackId >= levels.length || !track) {
      return;
    }
    this.log(`Subtitle track ${trackId} loaded [${newDetails.startSN},${newDetails.endSN}]${newDetails.lastPartSn ? `[part-${newDetails.lastPartSn}-${newDetails.lastPartIndex}]` : ""},duration:${newDetails.totalduration}`);
    this.mediaBuffer = this.mediaBufferTimeRanges;
    let sliding = 0;
    if (newDetails.live || (_track$details = track.details) != null && _track$details.live) {
      const mainDetails = this.mainDetails;
      if (newDetails.deltaUpdateFailed || !mainDetails) {
        return;
      }
      const mainSlidingStartFragment = mainDetails.fragments[0];
      if (!track.details) {
        if (newDetails.hasProgramDateTime && mainDetails.hasProgramDateTime) {
          alignMediaPlaylistByPDT(newDetails, mainDetails);
          sliding = newDetails.fragments[0].start;
        } else if (mainSlidingStartFragment) {
          sliding = mainSlidingStartFragment.start;
          addSliding(newDetails, sliding);
        }
      } else {
        var _this$levelLastLoaded;
        sliding = this.alignPlaylists(newDetails, track.details, (_this$levelLastLoaded = this.levelLastLoaded) == null ? void 0 : _this$levelLastLoaded.details);
        if (sliding === 0 && mainSlidingStartFragment) {
          sliding = mainSlidingStartFragment.start;
          addSliding(newDetails, sliding);
        }
      }
    }
    track.details = newDetails;
    this.levelLastLoaded = track;
    if (trackId !== currentTrackId) {
      return;
    }
    if (!this.startFragRequested && (this.mainDetails || !newDetails.live)) {
      this.setStartPosition(this.mainDetails || newDetails, sliding);
    }
    this.tick();
    if (newDetails.live && !this.fragCurrent && this.media && this.state === State.IDLE) {
      const foundFrag = findFragmentByPTS(null, newDetails.fragments, this.media.currentTime, 0);
      if (!foundFrag) {
        this.warn("Subtitle playlist not aligned with playback");
        track.details = void 0;
      }
    }
  }
  _handleFragmentLoadComplete(fragLoadedData) {
    const {
      frag,
      payload
    } = fragLoadedData;
    const decryptData = frag.decryptdata;
    const hls = this.hls;
    if (this.fragContextChanged(frag)) {
      return;
    }
    if (payload && payload.byteLength > 0 && decryptData != null && decryptData.key && decryptData.iv && decryptData.method === "AES-128") {
      const startTime = performance.now();
      this.decrypter.decrypt(new Uint8Array(payload), decryptData.key.buffer, decryptData.iv.buffer).catch((err) => {
        hls.trigger(Events.ERROR, {
          type: ErrorTypes.MEDIA_ERROR,
          details: ErrorDetails.FRAG_DECRYPT_ERROR,
          fatal: false,
          error: err,
          reason: err.message,
          frag
        });
        throw err;
      }).then((decryptedData) => {
        const endTime = performance.now();
        hls.trigger(Events.FRAG_DECRYPTED, {
          frag,
          payload: decryptedData,
          stats: {
            tstart: startTime,
            tdecrypt: endTime
          }
        });
      }).catch((err) => {
        this.warn(`${err.name}: ${err.message}`);
        this.state = State.IDLE;
      });
    }
  }
  doTick() {
    if (!this.media) {
      this.state = State.IDLE;
      return;
    }
    if (this.state === State.IDLE) {
      const {
        currentTrackId,
        levels
      } = this;
      const track = levels == null ? void 0 : levels[currentTrackId];
      if (!track || !levels.length || !track.details) {
        return;
      }
      const {
        config
      } = this;
      const currentTime = this.getLoadPosition();
      const bufferedInfo = BufferHelper.bufferedInfo(this.tracksBuffered[this.currentTrackId] || [], currentTime, config.maxBufferHole);
      const {
        end: targetBufferTime,
        len: bufferLen
      } = bufferedInfo;
      const mainBufferInfo = this.getFwdBufferInfo(this.media, PlaylistLevelType.MAIN);
      const trackDetails = track.details;
      const maxBufLen = this.getMaxBufferLength(mainBufferInfo == null ? void 0 : mainBufferInfo.len) + trackDetails.levelTargetDuration;
      if (bufferLen > maxBufLen) {
        return;
      }
      const fragments = trackDetails.fragments;
      const fragLen = fragments.length;
      const end = trackDetails.edge;
      let foundFrag = null;
      const fragPrevious = this.fragPrevious;
      if (targetBufferTime < end) {
        const tolerance = config.maxFragLookUpTolerance;
        const lookupTolerance = targetBufferTime > end - tolerance ? 0 : tolerance;
        foundFrag = findFragmentByPTS(fragPrevious, fragments, Math.max(fragments[0].start, targetBufferTime), lookupTolerance);
        if (!foundFrag && fragPrevious && fragPrevious.start < fragments[0].start) {
          foundFrag = fragments[0];
        }
      } else {
        foundFrag = fragments[fragLen - 1];
      }
      if (!foundFrag) {
        return;
      }
      foundFrag = this.mapToInitFragWhenRequired(foundFrag);
      if (foundFrag.sn !== "initSegment") {
        const curSNIdx = foundFrag.sn - trackDetails.startSN;
        const prevFrag = fragments[curSNIdx - 1];
        if (prevFrag && prevFrag.cc === foundFrag.cc && this.fragmentTracker.getState(prevFrag) === FragmentState.NOT_LOADED) {
          foundFrag = prevFrag;
        }
      }
      if (this.fragmentTracker.getState(foundFrag) === FragmentState.NOT_LOADED) {
        this.loadFragment(foundFrag, track, targetBufferTime);
      }
    }
  }
  getMaxBufferLength(mainBufferLength) {
    const maxConfigBuffer = super.getMaxBufferLength();
    if (!mainBufferLength) {
      return maxConfigBuffer;
    }
    return Math.max(maxConfigBuffer, mainBufferLength);
  }
  loadFragment(frag, level, targetBufferTime) {
    this.fragCurrent = frag;
    if (frag.sn === "initSegment") {
      this._loadInitSegment(frag, level);
    } else {
      this.startFragRequested = true;
      super.loadFragment(frag, level, targetBufferTime);
    }
  }
  get mediaBufferTimeRanges() {
    return new BufferableInstance(this.tracksBuffered[this.currentTrackId] || []);
  }
}
class BufferableInstance {
  constructor(timeranges) {
    this.buffered = void 0;
    const getRange = (name, index, length) => {
      index = index >>> 0;
      if (index > length - 1) {
        throw new DOMException(`Failed to execute '${name}' on 'TimeRanges': The index provided (${index}) is greater than the maximum bound (${length})`);
      }
      return timeranges[index][name];
    };
    this.buffered = {
      get length() {
        return timeranges.length;
      },
      end(index) {
        return getRange("end", index, timeranges.length);
      },
      start(index) {
        return getRange("start", index, timeranges.length);
      }
    };
  }
}
class SubtitleTrackController extends BasePlaylistController {
  constructor(hls) {
    super(hls, "[subtitle-track-controller]");
    this.media = null;
    this.tracks = [];
    this.groupIds = null;
    this.tracksInGroup = [];
    this.trackId = -1;
    this.currentTrack = null;
    this.selectDefaultTrack = true;
    this.queuedDefaultTrack = -1;
    this.asyncPollTrackChange = () => this.pollTrackChange(0);
    this.useTextTrackPolling = false;
    this.subtitlePollingInterval = -1;
    this._subtitleDisplay = true;
    this.onTextTracksChanged = () => {
      if (!this.useTextTrackPolling) {
        self.clearInterval(this.subtitlePollingInterval);
      }
      if (!this.media || !this.hls.config.renderTextTracksNatively) {
        return;
      }
      let textTrack = null;
      const tracks = filterSubtitleTracks(this.media.textTracks);
      for (let i2 = 0; i2 < tracks.length; i2++) {
        if (tracks[i2].mode === "hidden") {
          textTrack = tracks[i2];
        } else if (tracks[i2].mode === "showing") {
          textTrack = tracks[i2];
          break;
        }
      }
      const trackId = this.findTrackForTextTrack(textTrack);
      if (this.subtitleTrack !== trackId) {
        this.setSubtitleTrack(trackId);
      }
    };
    this.registerListeners();
  }
  destroy() {
    this.unregisterListeners();
    this.tracks.length = 0;
    this.tracksInGroup.length = 0;
    this.currentTrack = null;
    this.onTextTracksChanged = this.asyncPollTrackChange = null;
    super.destroy();
  }
  get subtitleDisplay() {
    return this._subtitleDisplay;
  }
  set subtitleDisplay(value) {
    this._subtitleDisplay = value;
    if (this.trackId > -1) {
      this.toggleTrackModes();
    }
  }
  registerListeners() {
    const {
      hls
    } = this;
    hls.on(Events.MEDIA_ATTACHED, this.onMediaAttached, this);
    hls.on(Events.MEDIA_DETACHING, this.onMediaDetaching, this);
    hls.on(Events.MANIFEST_LOADING, this.onManifestLoading, this);
    hls.on(Events.MANIFEST_PARSED, this.onManifestParsed, this);
    hls.on(Events.LEVEL_LOADING, this.onLevelLoading, this);
    hls.on(Events.LEVEL_SWITCHING, this.onLevelSwitching, this);
    hls.on(Events.SUBTITLE_TRACK_LOADED, this.onSubtitleTrackLoaded, this);
    hls.on(Events.ERROR, this.onError, this);
  }
  unregisterListeners() {
    const {
      hls
    } = this;
    hls.off(Events.MEDIA_ATTACHED, this.onMediaAttached, this);
    hls.off(Events.MEDIA_DETACHING, this.onMediaDetaching, this);
    hls.off(Events.MANIFEST_LOADING, this.onManifestLoading, this);
    hls.off(Events.MANIFEST_PARSED, this.onManifestParsed, this);
    hls.off(Events.LEVEL_LOADING, this.onLevelLoading, this);
    hls.off(Events.LEVEL_SWITCHING, this.onLevelSwitching, this);
    hls.off(Events.SUBTITLE_TRACK_LOADED, this.onSubtitleTrackLoaded, this);
    hls.off(Events.ERROR, this.onError, this);
  }
  // Listen for subtitle track change, then extract the current track ID.
  onMediaAttached(event, data) {
    this.media = data.media;
    if (!this.media) {
      return;
    }
    if (this.queuedDefaultTrack > -1) {
      this.subtitleTrack = this.queuedDefaultTrack;
      this.queuedDefaultTrack = -1;
    }
    this.useTextTrackPolling = !(this.media.textTracks && "onchange" in this.media.textTracks);
    if (this.useTextTrackPolling) {
      this.pollTrackChange(500);
    } else {
      this.media.textTracks.addEventListener("change", this.asyncPollTrackChange);
    }
  }
  pollTrackChange(timeout) {
    self.clearInterval(this.subtitlePollingInterval);
    this.subtitlePollingInterval = self.setInterval(this.onTextTracksChanged, timeout);
  }
  onMediaDetaching() {
    if (!this.media) {
      return;
    }
    self.clearInterval(this.subtitlePollingInterval);
    if (!this.useTextTrackPolling) {
      this.media.textTracks.removeEventListener("change", this.asyncPollTrackChange);
    }
    if (this.trackId > -1) {
      this.queuedDefaultTrack = this.trackId;
    }
    const textTracks = filterSubtitleTracks(this.media.textTracks);
    textTracks.forEach((track) => {
      clearCurrentCues(track);
    });
    this.subtitleTrack = -1;
    this.media = null;
  }
  onManifestLoading() {
    this.tracks = [];
    this.groupIds = null;
    this.tracksInGroup = [];
    this.trackId = -1;
    this.currentTrack = null;
    this.selectDefaultTrack = true;
  }
  // Fired whenever a new manifest is loaded.
  onManifestParsed(event, data) {
    this.tracks = data.subtitleTracks;
  }
  onSubtitleTrackLoaded(event, data) {
    const {
      id,
      groupId,
      details
    } = data;
    const trackInActiveGroup = this.tracksInGroup[id];
    if (!trackInActiveGroup || trackInActiveGroup.groupId !== groupId) {
      this.warn(`Subtitle track with id:${id} and group:${groupId} not found in active group ${trackInActiveGroup == null ? void 0 : trackInActiveGroup.groupId}`);
      return;
    }
    const curDetails = trackInActiveGroup.details;
    trackInActiveGroup.details = data.details;
    this.log(`Subtitle track ${id} "${trackInActiveGroup.name}" lang:${trackInActiveGroup.lang} group:${groupId} loaded [${details.startSN}-${details.endSN}]`);
    if (id === this.trackId) {
      this.playlistLoaded(id, data, curDetails);
    }
  }
  onLevelLoading(event, data) {
    this.switchLevel(data.level);
  }
  onLevelSwitching(event, data) {
    this.switchLevel(data.level);
  }
  switchLevel(levelIndex) {
    const levelInfo = this.hls.levels[levelIndex];
    if (!levelInfo) {
      return;
    }
    const subtitleGroups = levelInfo.subtitleGroups || null;
    const currentGroups = this.groupIds;
    let currentTrack = this.currentTrack;
    if (!subtitleGroups || (currentGroups == null ? void 0 : currentGroups.length) !== (subtitleGroups == null ? void 0 : subtitleGroups.length) || subtitleGroups != null && subtitleGroups.some((groupId) => (currentGroups == null ? void 0 : currentGroups.indexOf(groupId)) === -1)) {
      this.groupIds = subtitleGroups;
      this.trackId = -1;
      this.currentTrack = null;
      const subtitleTracks = this.tracks.filter((track) => !subtitleGroups || subtitleGroups.indexOf(track.groupId) !== -1);
      if (subtitleTracks.length) {
        if (this.selectDefaultTrack && !subtitleTracks.some((track) => track.default)) {
          this.selectDefaultTrack = false;
        }
        subtitleTracks.forEach((track, i2) => {
          track.id = i2;
        });
      } else if (!currentTrack && !this.tracksInGroup.length) {
        return;
      }
      this.tracksInGroup = subtitleTracks;
      const subtitlePreference = this.hls.config.subtitlePreference;
      if (!currentTrack && subtitlePreference) {
        this.selectDefaultTrack = false;
        const groupIndex = findMatchingOption(subtitlePreference, subtitleTracks);
        if (groupIndex > -1) {
          currentTrack = subtitleTracks[groupIndex];
        } else {
          const allIndex = findMatchingOption(subtitlePreference, this.tracks);
          currentTrack = this.tracks[allIndex];
        }
      }
      let trackId = this.findTrackId(currentTrack);
      if (trackId === -1 && currentTrack) {
        trackId = this.findTrackId(null);
      }
      const subtitleTracksUpdated = {
        subtitleTracks
      };
      this.log(`Updating subtitle tracks, ${subtitleTracks.length} track(s) found in "${subtitleGroups == null ? void 0 : subtitleGroups.join(",")}" group-id`);
      this.hls.trigger(Events.SUBTITLE_TRACKS_UPDATED, subtitleTracksUpdated);
      if (trackId !== -1 && this.trackId === -1) {
        this.setSubtitleTrack(trackId);
      }
    } else if (this.shouldReloadPlaylist(currentTrack)) {
      this.setSubtitleTrack(this.trackId);
    }
  }
  findTrackId(currentTrack) {
    const tracks = this.tracksInGroup;
    const selectDefault = this.selectDefaultTrack;
    for (let i2 = 0; i2 < tracks.length; i2++) {
      const track = tracks[i2];
      if (selectDefault && !track.default || !selectDefault && !currentTrack) {
        continue;
      }
      if (!currentTrack || matchesOption(track, currentTrack)) {
        return i2;
      }
    }
    if (currentTrack) {
      for (let i2 = 0; i2 < tracks.length; i2++) {
        const track = tracks[i2];
        if (mediaAttributesIdentical(currentTrack.attrs, track.attrs, ["LANGUAGE", "ASSOC-LANGUAGE", "CHARACTERISTICS"])) {
          return i2;
        }
      }
      for (let i2 = 0; i2 < tracks.length; i2++) {
        const track = tracks[i2];
        if (mediaAttributesIdentical(currentTrack.attrs, track.attrs, ["LANGUAGE"])) {
          return i2;
        }
      }
    }
    return -1;
  }
  findTrackForTextTrack(textTrack) {
    if (textTrack) {
      const tracks = this.tracksInGroup;
      for (let i2 = 0; i2 < tracks.length; i2++) {
        const track = tracks[i2];
        if (subtitleTrackMatchesTextTrack(track, textTrack)) {
          return i2;
        }
      }
    }
    return -1;
  }
  onError(event, data) {
    if (data.fatal || !data.context) {
      return;
    }
    if (data.context.type === PlaylistContextType.SUBTITLE_TRACK && data.context.id === this.trackId && (!this.groupIds || this.groupIds.indexOf(data.context.groupId) !== -1)) {
      this.checkRetry(data);
    }
  }
  get allSubtitleTracks() {
    return this.tracks;
  }
  /** get alternate subtitle tracks list from playlist **/
  get subtitleTracks() {
    return this.tracksInGroup;
  }
  /** get/set index of the selected subtitle track (based on index in subtitle track lists) **/
  get subtitleTrack() {
    return this.trackId;
  }
  set subtitleTrack(newId) {
    this.selectDefaultTrack = false;
    this.setSubtitleTrack(newId);
  }
  setSubtitleOption(subtitleOption) {
    this.hls.config.subtitlePreference = subtitleOption;
    if (subtitleOption) {
      const allSubtitleTracks = this.allSubtitleTracks;
      this.selectDefaultTrack = false;
      if (allSubtitleTracks.length) {
        const currentTrack = this.currentTrack;
        if (currentTrack && matchesOption(subtitleOption, currentTrack)) {
          return currentTrack;
        }
        const groupIndex = findMatchingOption(subtitleOption, this.tracksInGroup);
        if (groupIndex > -1) {
          const track = this.tracksInGroup[groupIndex];
          this.setSubtitleTrack(groupIndex);
          return track;
        } else if (currentTrack) {
          return null;
        } else {
          const allIndex = findMatchingOption(subtitleOption, allSubtitleTracks);
          if (allIndex > -1) {
            return allSubtitleTracks[allIndex];
          }
        }
      }
    }
    return null;
  }
  loadPlaylist(hlsUrlParameters) {
    super.loadPlaylist();
    const currentTrack = this.currentTrack;
    if (this.shouldLoadPlaylist(currentTrack) && currentTrack) {
      const id = currentTrack.id;
      const groupId = currentTrack.groupId;
      let url = currentTrack.url;
      if (hlsUrlParameters) {
        try {
          url = hlsUrlParameters.addDirectives(url);
        } catch (error) {
          this.warn(`Could not construct new URL with HLS Delivery Directives: ${error}`);
        }
      }
      this.log(`Loading subtitle playlist for id ${id}`);
      this.hls.trigger(Events.SUBTITLE_TRACK_LOADING, {
        url,
        id,
        groupId,
        deliveryDirectives: hlsUrlParameters || null
      });
    }
  }
  /**
   * Disables the old subtitleTrack and sets current mode on the next subtitleTrack.
   * This operates on the DOM textTracks.
   * A value of -1 will disable all subtitle tracks.
   */
  toggleTrackModes() {
    const {
      media
    } = this;
    if (!media) {
      return;
    }
    const textTracks = filterSubtitleTracks(media.textTracks);
    const currentTrack = this.currentTrack;
    let nextTrack;
    if (currentTrack) {
      nextTrack = textTracks.filter((textTrack) => subtitleTrackMatchesTextTrack(currentTrack, textTrack))[0];
      if (!nextTrack) {
        this.warn(`Unable to find subtitle TextTrack with name "${currentTrack.name}" and language "${currentTrack.lang}"`);
      }
    }
    [].slice.call(textTracks).forEach((track) => {
      if (track.mode !== "disabled" && track !== nextTrack) {
        track.mode = "disabled";
      }
    });
    if (nextTrack) {
      const mode = this.subtitleDisplay ? "showing" : "hidden";
      if (nextTrack.mode !== mode) {
        nextTrack.mode = mode;
      }
    }
  }
  /**
   * This method is responsible for validating the subtitle index and periodically reloading if live.
   * Dispatches the SUBTITLE_TRACK_SWITCH event, which instructs the subtitle-stream-controller to load the selected track.
   */
  setSubtitleTrack(newId) {
    const tracks = this.tracksInGroup;
    if (!this.media) {
      this.queuedDefaultTrack = newId;
      return;
    }
    if (newId < -1 || newId >= tracks.length || !isFiniteNumber(newId)) {
      this.warn(`Invalid subtitle track id: ${newId}`);
      return;
    }
    this.clearTimer();
    this.selectDefaultTrack = false;
    const lastTrack = this.currentTrack;
    const track = tracks[newId] || null;
    this.trackId = newId;
    this.currentTrack = track;
    this.toggleTrackModes();
    if (!track) {
      this.hls.trigger(Events.SUBTITLE_TRACK_SWITCH, {
        id: newId
      });
      return;
    }
    const trackLoaded = !!track.details && !track.details.live;
    if (newId === this.trackId && track === lastTrack && trackLoaded) {
      return;
    }
    this.log(`Switching to subtitle-track ${newId}` + (track ? ` "${track.name}" lang:${track.lang} group:${track.groupId}` : ""));
    const {
      id,
      groupId = "",
      name,
      type,
      url
    } = track;
    this.hls.trigger(Events.SUBTITLE_TRACK_SWITCH, {
      id,
      groupId,
      name,
      type,
      url
    });
    const hlsUrlParameters = this.switchParams(track.url, lastTrack == null ? void 0 : lastTrack.details, track.details);
    this.loadPlaylist(hlsUrlParameters);
  }
}
class BufferOperationQueue {
  constructor(sourceBufferReference) {
    this.buffers = void 0;
    this.queues = {
      video: [],
      audio: [],
      audiovideo: []
    };
    this.buffers = sourceBufferReference;
  }
  append(operation, type, pending) {
    const queue = this.queues[type];
    queue.push(operation);
    if (queue.length === 1 && !pending) {
      this.executeNext(type);
    }
  }
  insertAbort(operation, type) {
    const queue = this.queues[type];
    queue.unshift(operation);
    this.executeNext(type);
  }
  appendBlocker(type) {
    let execute;
    const promise = new Promise((resolve) => {
      execute = resolve;
    });
    const operation = {
      execute,
      onStart: () => {
      },
      onComplete: () => {
      },
      onError: () => {
      }
    };
    this.append(operation, type);
    return promise;
  }
  executeNext(type) {
    const queue = this.queues[type];
    if (queue.length) {
      const operation = queue[0];
      try {
        operation.execute();
      } catch (error) {
        logger.warn(`[buffer-operation-queue]: Exception executing "${type}" SourceBuffer operation: ${error}`);
        operation.onError(error);
        const sb = this.buffers[type];
        if (!(sb != null && sb.updating)) {
          this.shiftAndExecuteNext(type);
        }
      }
    }
  }
  shiftAndExecuteNext(type) {
    this.queues[type].shift();
    this.executeNext(type);
  }
  current(type) {
    return this.queues[type][0];
  }
}
const VIDEO_CODEC_PROFILE_REPLACE = /(avc[1234]|hvc1|hev1|dvh[1e]|vp09|av01)(?:\.[^.,]+)+/;
class BufferController {
  constructor(hls) {
    this.details = null;
    this._objectUrl = null;
    this.operationQueue = void 0;
    this.listeners = void 0;
    this.hls = void 0;
    this.bufferCodecEventsExpected = 0;
    this._bufferCodecEventsTotal = 0;
    this.media = null;
    this.mediaSource = null;
    this.lastMpegAudioChunk = null;
    this.appendSource = void 0;
    this.appendErrors = {
      audio: 0,
      video: 0,
      audiovideo: 0
    };
    this.tracks = {};
    this.pendingTracks = {};
    this.sourceBuffer = void 0;
    this.log = void 0;
    this.warn = void 0;
    this.error = void 0;
    this._onEndStreaming = (event) => {
      if (!this.hls) {
        return;
      }
      this.hls.pauseBuffering();
    };
    this._onStartStreaming = (event) => {
      if (!this.hls) {
        return;
      }
      this.hls.resumeBuffering();
    };
    this._onMediaSourceOpen = () => {
      const {
        media,
        mediaSource
      } = this;
      this.log("Media source opened");
      if (media) {
        media.removeEventListener("emptied", this._onMediaEmptied);
        this.updateMediaElementDuration();
        this.hls.trigger(Events.MEDIA_ATTACHED, {
          media,
          mediaSource
        });
      }
      if (mediaSource) {
        mediaSource.removeEventListener("sourceopen", this._onMediaSourceOpen);
      }
      this.checkPendingTracks();
    };
    this._onMediaSourceClose = () => {
      this.log("Media source closed");
    };
    this._onMediaSourceEnded = () => {
      this.log("Media source ended");
    };
    this._onMediaEmptied = () => {
      const {
        mediaSrc,
        _objectUrl
      } = this;
      if (mediaSrc !== _objectUrl) {
        logger.error(`Media element src was set while attaching MediaSource (${_objectUrl} > ${mediaSrc})`);
      }
    };
    this.hls = hls;
    const logPrefix = "[buffer-controller]";
    this.appendSource = isManagedMediaSource(getMediaSource(hls.config.preferManagedMediaSource));
    this.log = logger.log.bind(logger, logPrefix);
    this.warn = logger.warn.bind(logger, logPrefix);
    this.error = logger.error.bind(logger, logPrefix);
    this._initSourceBuffer();
    this.registerListeners();
  }
  hasSourceTypes() {
    return this.getSourceBufferTypes().length > 0 || Object.keys(this.pendingTracks).length > 0;
  }
  destroy() {
    this.unregisterListeners();
    this.details = null;
    this.lastMpegAudioChunk = null;
    this.hls = null;
  }
  registerListeners() {
    const {
      hls
    } = this;
    hls.on(Events.MEDIA_ATTACHING, this.onMediaAttaching, this);
    hls.on(Events.MEDIA_DETACHING, this.onMediaDetaching, this);
    hls.on(Events.MANIFEST_LOADING, this.onManifestLoading, this);
    hls.on(Events.MANIFEST_PARSED, this.onManifestParsed, this);
    hls.on(Events.BUFFER_RESET, this.onBufferReset, this);
    hls.on(Events.BUFFER_APPENDING, this.onBufferAppending, this);
    hls.on(Events.BUFFER_CODECS, this.onBufferCodecs, this);
    hls.on(Events.BUFFER_EOS, this.onBufferEos, this);
    hls.on(Events.BUFFER_FLUSHING, this.onBufferFlushing, this);
    hls.on(Events.LEVEL_UPDATED, this.onLevelUpdated, this);
    hls.on(Events.FRAG_PARSED, this.onFragParsed, this);
    hls.on(Events.FRAG_CHANGED, this.onFragChanged, this);
  }
  unregisterListeners() {
    const {
      hls
    } = this;
    hls.off(Events.MEDIA_ATTACHING, this.onMediaAttaching, this);
    hls.off(Events.MEDIA_DETACHING, this.onMediaDetaching, this);
    hls.off(Events.MANIFEST_LOADING, this.onManifestLoading, this);
    hls.off(Events.MANIFEST_PARSED, this.onManifestParsed, this);
    hls.off(Events.BUFFER_RESET, this.onBufferReset, this);
    hls.off(Events.BUFFER_APPENDING, this.onBufferAppending, this);
    hls.off(Events.BUFFER_CODECS, this.onBufferCodecs, this);
    hls.off(Events.BUFFER_EOS, this.onBufferEos, this);
    hls.off(Events.BUFFER_FLUSHING, this.onBufferFlushing, this);
    hls.off(Events.LEVEL_UPDATED, this.onLevelUpdated, this);
    hls.off(Events.FRAG_PARSED, this.onFragParsed, this);
    hls.off(Events.FRAG_CHANGED, this.onFragChanged, this);
  }
  _initSourceBuffer() {
    this.sourceBuffer = {};
    this.operationQueue = new BufferOperationQueue(this.sourceBuffer);
    this.listeners = {
      audio: [],
      video: [],
      audiovideo: []
    };
    this.appendErrors = {
      audio: 0,
      video: 0,
      audiovideo: 0
    };
    this.lastMpegAudioChunk = null;
  }
  onManifestLoading() {
    this.bufferCodecEventsExpected = this._bufferCodecEventsTotal = 0;
    this.details = null;
  }
  onManifestParsed(event, data) {
    let codecEvents = 2;
    if (data.audio && !data.video || !data.altAudio || false) {
      codecEvents = 1;
    }
    this.bufferCodecEventsExpected = this._bufferCodecEventsTotal = codecEvents;
    this.log(`${this.bufferCodecEventsExpected} bufferCodec event(s) expected`);
  }
  onMediaAttaching(event, data) {
    const media = this.media = data.media;
    const MediaSource = getMediaSource(this.appendSource);
    if (media && MediaSource) {
      var _ms$constructor;
      const ms = this.mediaSource = new MediaSource();
      this.log(`created media source: ${(_ms$constructor = ms.constructor) == null ? void 0 : _ms$constructor.name}`);
      ms.addEventListener("sourceopen", this._onMediaSourceOpen);
      ms.addEventListener("sourceended", this._onMediaSourceEnded);
      ms.addEventListener("sourceclose", this._onMediaSourceClose);
      if (this.appendSource) {
        ms.addEventListener("startstreaming", this._onStartStreaming);
        ms.addEventListener("endstreaming", this._onEndStreaming);
      }
      const objectUrl = this._objectUrl = self.URL.createObjectURL(ms);
      if (this.appendSource) {
        try {
          media.removeAttribute("src");
          const MMS = self.ManagedMediaSource;
          media.disableRemotePlayback = media.disableRemotePlayback || MMS && ms instanceof MMS;
          removeSourceChildren(media);
          addSource(media, objectUrl);
          media.load();
        } catch (error) {
          media.src = objectUrl;
        }
      } else {
        media.src = objectUrl;
      }
      media.addEventListener("emptied", this._onMediaEmptied);
    }
  }
  onMediaDetaching() {
    const {
      media,
      mediaSource,
      _objectUrl
    } = this;
    if (mediaSource) {
      this.log("media source detaching");
      if (mediaSource.readyState === "open") {
        try {
          mediaSource.endOfStream();
        } catch (err) {
          this.warn(`onMediaDetaching: ${err.message} while calling endOfStream`);
        }
      }
      this.onBufferReset();
      mediaSource.removeEventListener("sourceopen", this._onMediaSourceOpen);
      mediaSource.removeEventListener("sourceended", this._onMediaSourceEnded);
      mediaSource.removeEventListener("sourceclose", this._onMediaSourceClose);
      if (this.appendSource) {
        mediaSource.removeEventListener("startstreaming", this._onStartStreaming);
        mediaSource.removeEventListener("endstreaming", this._onEndStreaming);
      }
      if (media) {
        media.removeEventListener("emptied", this._onMediaEmptied);
        if (_objectUrl) {
          self.URL.revokeObjectURL(_objectUrl);
        }
        if (this.mediaSrc === _objectUrl) {
          media.removeAttribute("src");
          if (this.appendSource) {
            removeSourceChildren(media);
          }
          media.load();
        } else {
          this.warn("media|source.src was changed by a third party - skip cleanup");
        }
      }
      this.mediaSource = null;
      this.media = null;
      this._objectUrl = null;
      this.bufferCodecEventsExpected = this._bufferCodecEventsTotal;
      this.pendingTracks = {};
      this.tracks = {};
    }
    this.hls.trigger(Events.MEDIA_DETACHED, void 0);
  }
  onBufferReset() {
    this.getSourceBufferTypes().forEach((type) => {
      this.resetBuffer(type);
    });
    this._initSourceBuffer();
  }
  resetBuffer(type) {
    const sb = this.sourceBuffer[type];
    try {
      if (sb) {
        var _this$mediaSource;
        this.removeBufferListeners(type);
        this.sourceBuffer[type] = void 0;
        if ((_this$mediaSource = this.mediaSource) != null && _this$mediaSource.sourceBuffers.length) {
          this.mediaSource.removeSourceBuffer(sb);
        }
      }
    } catch (err) {
      this.warn(`onBufferReset ${type}`, err);
    }
  }
  onBufferCodecs(event, data) {
    const sourceBufferCount = this.getSourceBufferTypes().length;
    const trackNames = Object.keys(data);
    trackNames.forEach((trackName) => {
      if (sourceBufferCount) {
        const track = this.tracks[trackName];
        if (track && typeof track.buffer.changeType === "function") {
          var _trackCodec;
          const {
            id,
            codec,
            levelCodec,
            container,
            metadata
          } = data[trackName];
          const currentCodecFull = pickMostCompleteCodecName(track.codec, track.levelCodec);
          const currentCodec = currentCodecFull == null ? void 0 : currentCodecFull.replace(VIDEO_CODEC_PROFILE_REPLACE, "$1");
          let trackCodec = pickMostCompleteCodecName(codec, levelCodec);
          const nextCodec = (_trackCodec = trackCodec) == null ? void 0 : _trackCodec.replace(VIDEO_CODEC_PROFILE_REPLACE, "$1");
          if (trackCodec && currentCodec !== nextCodec) {
            if (trackName.slice(0, 5) === "audio") {
              trackCodec = getCodecCompatibleName(trackCodec, this.appendSource);
            }
            const mimeType = `${container};codecs=${trackCodec}`;
            this.appendChangeType(trackName, mimeType);
            this.log(`switching codec ${currentCodecFull} to ${trackCodec}`);
            this.tracks[trackName] = {
              buffer: track.buffer,
              codec,
              container,
              levelCodec,
              metadata,
              id
            };
          }
        }
      } else {
        this.pendingTracks[trackName] = data[trackName];
      }
    });
    if (sourceBufferCount) {
      return;
    }
    const bufferCodecEventsExpected = Math.max(this.bufferCodecEventsExpected - 1, 0);
    if (this.bufferCodecEventsExpected !== bufferCodecEventsExpected) {
      this.log(`${bufferCodecEventsExpected} bufferCodec event(s) expected ${trackNames.join(",")}`);
      this.bufferCodecEventsExpected = bufferCodecEventsExpected;
    }
    if (this.mediaSource && this.mediaSource.readyState === "open") {
      this.checkPendingTracks();
    }
  }
  appendChangeType(type, mimeType) {
    const {
      operationQueue
    } = this;
    const operation = {
      execute: () => {
        const sb = this.sourceBuffer[type];
        if (sb) {
          this.log(`changing ${type} sourceBuffer type to ${mimeType}`);
          sb.changeType(mimeType);
        }
        operationQueue.shiftAndExecuteNext(type);
      },
      onStart: () => {
      },
      onComplete: () => {
      },
      onError: (error) => {
        this.warn(`Failed to change ${type} SourceBuffer type`, error);
      }
    };
    operationQueue.append(operation, type, !!this.pendingTracks[type]);
  }
  onBufferAppending(event, eventData) {
    const {
      hls,
      operationQueue,
      tracks
    } = this;
    const {
      data,
      type,
      frag,
      part,
      chunkMeta
    } = eventData;
    const chunkStats = chunkMeta.buffering[type];
    const bufferAppendingStart = self.performance.now();
    chunkStats.start = bufferAppendingStart;
    const fragBuffering = frag.stats.buffering;
    const partBuffering = part ? part.stats.buffering : null;
    if (fragBuffering.start === 0) {
      fragBuffering.start = bufferAppendingStart;
    }
    if (partBuffering && partBuffering.start === 0) {
      partBuffering.start = bufferAppendingStart;
    }
    const audioTrack = tracks.audio;
    let checkTimestampOffset = false;
    if (type === "audio" && (audioTrack == null ? void 0 : audioTrack.container) === "audio/mpeg") {
      checkTimestampOffset = !this.lastMpegAudioChunk || chunkMeta.id === 1 || this.lastMpegAudioChunk.sn !== chunkMeta.sn;
      this.lastMpegAudioChunk = chunkMeta;
    }
    const fragStart = frag.start;
    const operation = {
      execute: () => {
        chunkStats.executeStart = self.performance.now();
        if (checkTimestampOffset) {
          const sb = this.sourceBuffer[type];
          if (sb) {
            const delta = fragStart - sb.timestampOffset;
            if (Math.abs(delta) >= 0.1) {
              this.log(`Updating audio SourceBuffer timestampOffset to ${fragStart} (delta: ${delta}) sn: ${frag.sn})`);
              sb.timestampOffset = fragStart;
            }
          }
        }
        this.appendExecutor(data, type);
      },
      onStart: () => {
      },
      onComplete: () => {
        const end = self.performance.now();
        chunkStats.executeEnd = chunkStats.end = end;
        if (fragBuffering.first === 0) {
          fragBuffering.first = end;
        }
        if (partBuffering && partBuffering.first === 0) {
          partBuffering.first = end;
        }
        const {
          sourceBuffer
        } = this;
        const timeRanges = {};
        for (const type2 in sourceBuffer) {
          timeRanges[type2] = BufferHelper.getBuffered(sourceBuffer[type2]);
        }
        this.appendErrors[type] = 0;
        if (type === "audio" || type === "video") {
          this.appendErrors.audiovideo = 0;
        } else {
          this.appendErrors.audio = 0;
          this.appendErrors.video = 0;
        }
        this.hls.trigger(Events.BUFFER_APPENDED, {
          type,
          frag,
          part,
          chunkMeta,
          parent: frag.type,
          timeRanges
        });
      },
      onError: (error) => {
        const event2 = {
          type: ErrorTypes.MEDIA_ERROR,
          parent: frag.type,
          details: ErrorDetails.BUFFER_APPEND_ERROR,
          sourceBufferName: type,
          frag,
          part,
          chunkMeta,
          error,
          err: error,
          fatal: false
        };
        if (error.code === DOMException.QUOTA_EXCEEDED_ERR) {
          event2.details = ErrorDetails.BUFFER_FULL_ERROR;
        } else {
          const appendErrorCount = ++this.appendErrors[type];
          event2.details = ErrorDetails.BUFFER_APPEND_ERROR;
          this.warn(`Failed ${appendErrorCount}/${hls.config.appendErrorMaxRetry} times to append segment in "${type}" sourceBuffer`);
          if (appendErrorCount >= hls.config.appendErrorMaxRetry) {
            event2.fatal = true;
          }
        }
        hls.trigger(Events.ERROR, event2);
      }
    };
    operationQueue.append(operation, type, !!this.pendingTracks[type]);
  }
  onBufferFlushing(event, data) {
    const {
      operationQueue
    } = this;
    const flushOperation = (type) => ({
      execute: this.removeExecutor.bind(this, type, data.startOffset, data.endOffset),
      onStart: () => {
      },
      onComplete: () => {
        this.hls.trigger(Events.BUFFER_FLUSHED, {
          type
        });
      },
      onError: (error) => {
        this.warn(`Failed to remove from ${type} SourceBuffer`, error);
      }
    });
    if (data.type) {
      operationQueue.append(flushOperation(data.type), data.type);
    } else {
      this.getSourceBufferTypes().forEach((type) => {
        operationQueue.append(flushOperation(type), type);
      });
    }
  }
  onFragParsed(event, data) {
    const {
      frag,
      part
    } = data;
    const buffersAppendedTo = [];
    const elementaryStreams = part ? part.elementaryStreams : frag.elementaryStreams;
    if (elementaryStreams[ElementaryStreamTypes.AUDIOVIDEO]) {
      buffersAppendedTo.push("audiovideo");
    } else {
      if (elementaryStreams[ElementaryStreamTypes.AUDIO]) {
        buffersAppendedTo.push("audio");
      }
      if (elementaryStreams[ElementaryStreamTypes.VIDEO]) {
        buffersAppendedTo.push("video");
      }
    }
    const onUnblocked = () => {
      const now2 = self.performance.now();
      frag.stats.buffering.end = now2;
      if (part) {
        part.stats.buffering.end = now2;
      }
      const stats = part ? part.stats : frag.stats;
      this.hls.trigger(Events.FRAG_BUFFERED, {
        frag,
        part,
        stats,
        id: frag.type
      });
    };
    if (buffersAppendedTo.length === 0) {
      this.warn(`Fragments must have at least one ElementaryStreamType set. type: ${frag.type} level: ${frag.level} sn: ${frag.sn}`);
    }
    this.blockBuffers(onUnblocked, buffersAppendedTo);
  }
  onFragChanged(event, data) {
    this.trimBuffers();
  }
  // on BUFFER_EOS mark matching sourcebuffer(s) as ended and trigger checkEos()
  // an undefined data.type will mark all buffers as EOS.
  onBufferEos(event, data) {
    const ended = this.getSourceBufferTypes().reduce((acc, type) => {
      const sb = this.sourceBuffer[type];
      if (sb && (!data.type || data.type === type)) {
        sb.ending = true;
        if (!sb.ended) {
          sb.ended = true;
          this.log(`${type} sourceBuffer now EOS`);
        }
      }
      return acc && !!(!sb || sb.ended);
    }, true);
    if (ended) {
      this.log(`Queueing mediaSource.endOfStream()`);
      this.blockBuffers(() => {
        this.getSourceBufferTypes().forEach((type) => {
          const sb = this.sourceBuffer[type];
          if (sb) {
            sb.ending = false;
          }
        });
        const {
          mediaSource
        } = this;
        if (!mediaSource || mediaSource.readyState !== "open") {
          if (mediaSource) {
            this.log(`Could not call mediaSource.endOfStream(). mediaSource.readyState: ${mediaSource.readyState}`);
          }
          return;
        }
        this.log(`Calling mediaSource.endOfStream()`);
        mediaSource.endOfStream();
      });
    }
  }
  onLevelUpdated(event, {
    details
  }) {
    if (!details.fragments.length) {
      return;
    }
    this.details = details;
    if (this.getSourceBufferTypes().length) {
      this.blockBuffers(this.updateMediaElementDuration.bind(this));
    } else {
      this.updateMediaElementDuration();
    }
  }
  trimBuffers() {
    const {
      hls,
      details,
      media
    } = this;
    if (!media || details === null) {
      return;
    }
    const sourceBufferTypes = this.getSourceBufferTypes();
    if (!sourceBufferTypes.length) {
      return;
    }
    const config = hls.config;
    const currentTime = media.currentTime;
    const targetDuration = details.levelTargetDuration;
    const backBufferLength = details.live && config.liveBackBufferLength !== null ? config.liveBackBufferLength : config.backBufferLength;
    if (isFiniteNumber(backBufferLength) && backBufferLength > 0) {
      const maxBackBufferLength = Math.max(backBufferLength, targetDuration);
      const targetBackBufferPosition = Math.floor(currentTime / targetDuration) * targetDuration - maxBackBufferLength;
      this.flushBackBuffer(currentTime, targetDuration, targetBackBufferPosition);
    }
    if (isFiniteNumber(config.frontBufferFlushThreshold) && config.frontBufferFlushThreshold > 0) {
      const frontBufferLength = Math.max(config.maxBufferLength, config.frontBufferFlushThreshold);
      const maxFrontBufferLength = Math.max(frontBufferLength, targetDuration);
      const targetFrontBufferPosition = Math.floor(currentTime / targetDuration) * targetDuration + maxFrontBufferLength;
      this.flushFrontBuffer(currentTime, targetDuration, targetFrontBufferPosition);
    }
  }
  flushBackBuffer(currentTime, targetDuration, targetBackBufferPosition) {
    const {
      details,
      sourceBuffer
    } = this;
    const sourceBufferTypes = this.getSourceBufferTypes();
    sourceBufferTypes.forEach((type) => {
      const sb = sourceBuffer[type];
      if (sb) {
        const buffered = BufferHelper.getBuffered(sb);
        if (buffered.length > 0 && targetBackBufferPosition > buffered.start(0)) {
          this.hls.trigger(Events.BACK_BUFFER_REACHED, {
            bufferEnd: targetBackBufferPosition
          });
          if (details != null && details.live) {
            this.hls.trigger(Events.LIVE_BACK_BUFFER_REACHED, {
              bufferEnd: targetBackBufferPosition
            });
          } else if (sb.ended && buffered.end(buffered.length - 1) - currentTime < targetDuration * 2) {
            this.log(`Cannot flush ${type} back buffer while SourceBuffer is in ended state`);
            return;
          }
          this.hls.trigger(Events.BUFFER_FLUSHING, {
            startOffset: 0,
            endOffset: targetBackBufferPosition,
            type
          });
        }
      }
    });
  }
  flushFrontBuffer(currentTime, targetDuration, targetFrontBufferPosition) {
    const {
      sourceBuffer
    } = this;
    const sourceBufferTypes = this.getSourceBufferTypes();
    sourceBufferTypes.forEach((type) => {
      const sb = sourceBuffer[type];
      if (sb) {
        const buffered = BufferHelper.getBuffered(sb);
        const numBufferedRanges = buffered.length;
        if (numBufferedRanges < 2) {
          return;
        }
        const bufferStart = buffered.start(numBufferedRanges - 1);
        const bufferEnd = buffered.end(numBufferedRanges - 1);
        if (targetFrontBufferPosition > bufferStart || currentTime >= bufferStart && currentTime <= bufferEnd) {
          return;
        } else if (sb.ended && currentTime - bufferEnd < 2 * targetDuration) {
          this.log(`Cannot flush ${type} front buffer while SourceBuffer is in ended state`);
          return;
        }
        this.hls.trigger(Events.BUFFER_FLUSHING, {
          startOffset: bufferStart,
          endOffset: Infinity,
          type
        });
      }
    });
  }
  /**
   * Update Media Source duration to current level duration or override to Infinity if configuration parameter
   * 'liveDurationInfinity` is set to `true`
   * More details: https://github.com/video-dev/hls.js/issues/355
   */
  updateMediaElementDuration() {
    if (!this.details || !this.media || !this.mediaSource || this.mediaSource.readyState !== "open") {
      return;
    }
    const {
      details,
      hls,
      media,
      mediaSource
    } = this;
    const levelDuration = details.fragments[0].start + details.totalduration;
    const mediaDuration = media.duration;
    const msDuration = isFiniteNumber(mediaSource.duration) ? mediaSource.duration : 0;
    if (details.live && hls.config.liveDurationInfinity) {
      mediaSource.duration = Infinity;
      this.updateSeekableRange(details);
    } else if (levelDuration > msDuration && levelDuration > mediaDuration || !isFiniteNumber(mediaDuration)) {
      this.log(`Updating Media Source duration to ${levelDuration.toFixed(3)}`);
      mediaSource.duration = levelDuration;
    }
  }
  updateSeekableRange(levelDetails) {
    const mediaSource = this.mediaSource;
    const fragments = levelDetails.fragments;
    const len = fragments.length;
    if (len && levelDetails.live && mediaSource != null && mediaSource.setLiveSeekableRange) {
      const start = Math.max(0, fragments[0].start);
      const end = Math.max(start, start + levelDetails.totalduration);
      this.log(`Media Source duration is set to ${mediaSource.duration}. Setting seekable range to ${start}-${end}.`);
      mediaSource.setLiveSeekableRange(start, end);
    }
  }
  checkPendingTracks() {
    const {
      bufferCodecEventsExpected,
      operationQueue,
      pendingTracks
    } = this;
    const pendingTracksCount = Object.keys(pendingTracks).length;
    if (pendingTracksCount && (!bufferCodecEventsExpected || pendingTracksCount === 2 || "audiovideo" in pendingTracks)) {
      this.createSourceBuffers(pendingTracks);
      this.pendingTracks = {};
      const buffers = this.getSourceBufferTypes();
      if (buffers.length) {
        this.hls.trigger(Events.BUFFER_CREATED, {
          tracks: this.tracks
        });
        buffers.forEach((type) => {
          operationQueue.executeNext(type);
        });
      } else {
        const error = new Error("could not create source buffer for media codec(s)");
        this.hls.trigger(Events.ERROR, {
          type: ErrorTypes.MEDIA_ERROR,
          details: ErrorDetails.BUFFER_INCOMPATIBLE_CODECS_ERROR,
          fatal: true,
          error,
          reason: error.message
        });
      }
    }
  }
  createSourceBuffers(tracks) {
    const {
      sourceBuffer,
      mediaSource
    } = this;
    if (!mediaSource) {
      throw Error("createSourceBuffers called when mediaSource was null");
    }
    for (const trackName in tracks) {
      if (!sourceBuffer[trackName]) {
        var _track$levelCodec;
        const track = tracks[trackName];
        if (!track) {
          throw Error(`source buffer exists for track ${trackName}, however track does not`);
        }
        let codec = ((_track$levelCodec = track.levelCodec) == null ? void 0 : _track$levelCodec.indexOf(",")) === -1 ? track.levelCodec : track.codec;
        if (codec) {
          if (trackName.slice(0, 5) === "audio") {
            codec = getCodecCompatibleName(codec, this.appendSource);
          }
        }
        const mimeType = `${track.container};codecs=${codec}`;
        this.log(`creating sourceBuffer(${mimeType})`);
        try {
          const sb = sourceBuffer[trackName] = mediaSource.addSourceBuffer(mimeType);
          const sbName = trackName;
          this.addBufferListener(sbName, "updatestart", this._onSBUpdateStart);
          this.addBufferListener(sbName, "updateend", this._onSBUpdateEnd);
          this.addBufferListener(sbName, "error", this._onSBUpdateError);
          if (this.appendSource) {
            this.addBufferListener(sbName, "bufferedchange", (type, event) => {
              const removedRanges = event.removedRanges;
              if (removedRanges != null && removedRanges.length) {
                this.hls.trigger(Events.BUFFER_FLUSHED, {
                  type: trackName
                });
              }
            });
          }
          this.tracks[trackName] = {
            buffer: sb,
            codec,
            container: track.container,
            levelCodec: track.levelCodec,
            metadata: track.metadata,
            id: track.id
          };
        } catch (err) {
          this.error(`error while trying to add sourceBuffer: ${err.message}`);
          this.hls.trigger(Events.ERROR, {
            type: ErrorTypes.MEDIA_ERROR,
            details: ErrorDetails.BUFFER_ADD_CODEC_ERROR,
            fatal: false,
            error: err,
            sourceBufferName: trackName,
            mimeType
          });
        }
      }
    }
  }
  get mediaSrc() {
    var _this$media;
    const media = ((_this$media = this.media) == null ? void 0 : _this$media.firstChild) || this.media;
    return media == null ? void 0 : media.src;
  }
  _onSBUpdateStart(type) {
    const {
      operationQueue
    } = this;
    const operation = operationQueue.current(type);
    operation.onStart();
  }
  _onSBUpdateEnd(type) {
    var _this$mediaSource2;
    if (((_this$mediaSource2 = this.mediaSource) == null ? void 0 : _this$mediaSource2.readyState) === "closed") {
      this.resetBuffer(type);
      return;
    }
    const {
      operationQueue
    } = this;
    const operation = operationQueue.current(type);
    operation.onComplete();
    operationQueue.shiftAndExecuteNext(type);
  }
  _onSBUpdateError(type, event) {
    var _this$mediaSource3;
    const error = new Error(`${type} SourceBuffer error. MediaSource readyState: ${(_this$mediaSource3 = this.mediaSource) == null ? void 0 : _this$mediaSource3.readyState}`);
    this.error(`${error}`, event);
    this.hls.trigger(Events.ERROR, {
      type: ErrorTypes.MEDIA_ERROR,
      details: ErrorDetails.BUFFER_APPENDING_ERROR,
      sourceBufferName: type,
      error,
      fatal: false
    });
    const operation = this.operationQueue.current(type);
    if (operation) {
      operation.onError(error);
    }
  }
  // This method must result in an updateend event; if remove is not called, _onSBUpdateEnd must be called manually
  removeExecutor(type, startOffset, endOffset) {
    const {
      media,
      mediaSource,
      operationQueue,
      sourceBuffer
    } = this;
    const sb = sourceBuffer[type];
    if (!media || !mediaSource || !sb) {
      this.warn(`Attempting to remove from the ${type} SourceBuffer, but it does not exist`);
      operationQueue.shiftAndExecuteNext(type);
      return;
    }
    const mediaDuration = isFiniteNumber(media.duration) ? media.duration : Infinity;
    const msDuration = isFiniteNumber(mediaSource.duration) ? mediaSource.duration : Infinity;
    const removeStart = Math.max(0, startOffset);
    const removeEnd = Math.min(endOffset, mediaDuration, msDuration);
    if (removeEnd > removeStart && (!sb.ending || sb.ended)) {
      sb.ended = false;
      this.log(`Removing [${removeStart},${removeEnd}] from the ${type} SourceBuffer`);
      sb.remove(removeStart, removeEnd);
    } else {
      operationQueue.shiftAndExecuteNext(type);
    }
  }
  // This method must result in an updateend event; if append is not called, _onSBUpdateEnd must be called manually
  appendExecutor(data, type) {
    const sb = this.sourceBuffer[type];
    if (!sb) {
      if (!this.pendingTracks[type]) {
        throw new Error(`Attempting to append to the ${type} SourceBuffer, but it does not exist`);
      }
      return;
    }
    sb.ended = false;
    sb.appendBuffer(data);
  }
  // Enqueues an operation to each SourceBuffer queue which, upon execution, resolves a promise. When all promises
  // resolve, the onUnblocked function is executed. Functions calling this method do not need to unblock the queue
  // upon completion, since we already do it here
  blockBuffers(onUnblocked, buffers = this.getSourceBufferTypes()) {
    if (!buffers.length) {
      this.log("Blocking operation requested, but no SourceBuffers exist");
      Promise.resolve().then(onUnblocked);
      return;
    }
    const {
      operationQueue
    } = this;
    const blockingOperations = buffers.map((type) => operationQueue.appendBlocker(type));
    Promise.all(blockingOperations).then(() => {
      onUnblocked();
      buffers.forEach((type) => {
        const sb = this.sourceBuffer[type];
        if (!(sb != null && sb.updating)) {
          operationQueue.shiftAndExecuteNext(type);
        }
      });
    });
  }
  getSourceBufferTypes() {
    return Object.keys(this.sourceBuffer);
  }
  addBufferListener(type, event, fn) {
    const buffer = this.sourceBuffer[type];
    if (!buffer) {
      return;
    }
    const listener = fn.bind(this, type);
    this.listeners[type].push({
      event,
      listener
    });
    buffer.addEventListener(event, listener);
  }
  removeBufferListeners(type) {
    const buffer = this.sourceBuffer[type];
    if (!buffer) {
      return;
    }
    this.listeners[type].forEach((l) => {
      buffer.removeEventListener(l.event, l.listener);
    });
  }
}
function removeSourceChildren(node) {
  const sourceChildren = node.querySelectorAll("source");
  [].slice.call(sourceChildren).forEach((source) => {
    node.removeChild(source);
  });
}
function addSource(media, url) {
  const source = self.document.createElement("source");
  source.type = "video/mp4";
  source.src = url;
  media.appendChild(source);
}
const specialCea608CharsCodes = {
  42: 225,
  // lowercase a, acute accent
  92: 233,
  // lowercase e, acute accent
  94: 237,
  // lowercase i, acute accent
  95: 243,
  // lowercase o, acute accent
  96: 250,
  // lowercase u, acute accent
  123: 231,
  // lowercase c with cedilla
  124: 247,
  // division symbol
  125: 209,
  // uppercase N tilde
  126: 241,
  // lowercase n tilde
  127: 9608,
  // Full block
  // THIS BLOCK INCLUDES THE 16 EXTENDED (TWO-BYTE) LINE 21 CHARACTERS
  // THAT COME FROM HI BYTE=0x11 AND LOW BETWEEN 0x30 AND 0x3F
  // THIS MEANS THAT \x50 MUST BE ADDED TO THE VALUES
  128: 174,
  // Registered symbol (R)
  129: 176,
  // degree sign
  130: 189,
  // 1/2 symbol
  131: 191,
  // Inverted (open) question mark
  132: 8482,
  // Trademark symbol (TM)
  133: 162,
  // Cents symbol
  134: 163,
  // Pounds sterling
  135: 9834,
  // Music 8'th note
  136: 224,
  // lowercase a, grave accent
  137: 32,
  // transparent space (regular)
  138: 232,
  // lowercase e, grave accent
  139: 226,
  // lowercase a, circumflex accent
  140: 234,
  // lowercase e, circumflex accent
  141: 238,
  // lowercase i, circumflex accent
  142: 244,
  // lowercase o, circumflex accent
  143: 251,
  // lowercase u, circumflex accent
  // THIS BLOCK INCLUDES THE 32 EXTENDED (TWO-BYTE) LINE 21 CHARACTERS
  // THAT COME FROM HI BYTE=0x12 AND LOW BETWEEN 0x20 AND 0x3F
  144: 193,
  // capital letter A with acute
  145: 201,
  // capital letter E with acute
  146: 211,
  // capital letter O with acute
  147: 218,
  // capital letter U with acute
  148: 220,
  // capital letter U with diaresis
  149: 252,
  // lowercase letter U with diaeresis
  150: 8216,
  // opening single quote
  151: 161,
  // inverted exclamation mark
  152: 42,
  // asterisk
  153: 8217,
  // closing single quote
  154: 9473,
  // box drawings heavy horizontal
  155: 169,
  // copyright sign
  156: 8480,
  // Service mark
  157: 8226,
  // (round) bullet
  158: 8220,
  // Left double quotation mark
  159: 8221,
  // Right double quotation mark
  160: 192,
  // uppercase A, grave accent
  161: 194,
  // uppercase A, circumflex
  162: 199,
  // uppercase C with cedilla
  163: 200,
  // uppercase E, grave accent
  164: 202,
  // uppercase E, circumflex
  165: 203,
  // capital letter E with diaresis
  166: 235,
  // lowercase letter e with diaresis
  167: 206,
  // uppercase I, circumflex
  168: 207,
  // uppercase I, with diaresis
  169: 239,
  // lowercase i, with diaresis
  170: 212,
  // uppercase O, circumflex
  171: 217,
  // uppercase U, grave accent
  172: 249,
  // lowercase u, grave accent
  173: 219,
  // uppercase U, circumflex
  174: 171,
  // left-pointing double angle quotation mark
  175: 187,
  // right-pointing double angle quotation mark
  // THIS BLOCK INCLUDES THE 32 EXTENDED (TWO-BYTE) LINE 21 CHARACTERS
  // THAT COME FROM HI BYTE=0x13 AND LOW BETWEEN 0x20 AND 0x3F
  176: 195,
  // Uppercase A, tilde
  177: 227,
  // Lowercase a, tilde
  178: 205,
  // Uppercase I, acute accent
  179: 204,
  // Uppercase I, grave accent
  180: 236,
  // Lowercase i, grave accent
  181: 210,
  // Uppercase O, grave accent
  182: 242,
  // Lowercase o, grave accent
  183: 213,
  // Uppercase O, tilde
  184: 245,
  // Lowercase o, tilde
  185: 123,
  // Open curly brace
  186: 125,
  // Closing curly brace
  187: 92,
  // Backslash
  188: 94,
  // Caret
  189: 95,
  // Underscore
  190: 124,
  // Pipe (vertical line)
  191: 8764,
  // Tilde operator
  192: 196,
  // Uppercase A, umlaut
  193: 228,
  // Lowercase A, umlaut
  194: 214,
  // Uppercase O, umlaut
  195: 246,
  // Lowercase o, umlaut
  196: 223,
  // Esszett (sharp S)
  197: 165,
  // Yen symbol
  198: 164,
  // Generic currency sign
  199: 9475,
  // Box drawings heavy vertical
  200: 197,
  // Uppercase A, ring
  201: 229,
  // Lowercase A, ring
  202: 216,
  // Uppercase O, stroke
  203: 248,
  // Lowercase o, strok
  204: 9487,
  // Box drawings heavy down and right
  205: 9491,
  // Box drawings heavy down and left
  206: 9495,
  // Box drawings heavy up and right
  207: 9499
  // Box drawings heavy up and left
};
const getCharForByte = (byte) => String.fromCharCode(specialCea608CharsCodes[byte] || byte);
const NR_ROWS = 15;
const NR_COLS = 100;
const rowsLowCh1 = {
  17: 1,
  18: 3,
  21: 5,
  22: 7,
  23: 9,
  16: 11,
  19: 12,
  20: 14
};
const rowsHighCh1 = {
  17: 2,
  18: 4,
  21: 6,
  22: 8,
  23: 10,
  19: 13,
  20: 15
};
const rowsLowCh2 = {
  25: 1,
  26: 3,
  29: 5,
  30: 7,
  31: 9,
  24: 11,
  27: 12,
  28: 14
};
const rowsHighCh2 = {
  25: 2,
  26: 4,
  29: 6,
  30: 8,
  31: 10,
  27: 13,
  28: 15
};
const backgroundColors = ["white", "green", "blue", "cyan", "red", "yellow", "magenta", "black", "transparent"];
class CaptionsLogger {
  constructor() {
    this.time = null;
    this.verboseLevel = 0;
  }
  log(severity, msg) {
    if (this.verboseLevel >= severity) {
      const m = typeof msg === "function" ? msg() : msg;
      logger.log(`${this.time} [${severity}] ${m}`);
    }
  }
}
const numArrayToHexArray = function numArrayToHexArray2(numArray) {
  const hexArray = [];
  for (let j = 0; j < numArray.length; j++) {
    hexArray.push(numArray[j].toString(16));
  }
  return hexArray;
};
class PenState {
  constructor() {
    this.foreground = "white";
    this.underline = false;
    this.italics = false;
    this.background = "black";
    this.flash = false;
  }
  reset() {
    this.foreground = "white";
    this.underline = false;
    this.italics = false;
    this.background = "black";
    this.flash = false;
  }
  setStyles(styles) {
    const attribs = ["foreground", "underline", "italics", "background", "flash"];
    for (let i2 = 0; i2 < attribs.length; i2++) {
      const style = attribs[i2];
      if (styles.hasOwnProperty(style)) {
        this[style] = styles[style];
      }
    }
  }
  isDefault() {
    return this.foreground === "white" && !this.underline && !this.italics && this.background === "black" && !this.flash;
  }
  equals(other) {
    return this.foreground === other.foreground && this.underline === other.underline && this.italics === other.italics && this.background === other.background && this.flash === other.flash;
  }
  copy(newPenState) {
    this.foreground = newPenState.foreground;
    this.underline = newPenState.underline;
    this.italics = newPenState.italics;
    this.background = newPenState.background;
    this.flash = newPenState.flash;
  }
  toString() {
    return "color=" + this.foreground + ", underline=" + this.underline + ", italics=" + this.italics + ", background=" + this.background + ", flash=" + this.flash;
  }
}
class StyledUnicodeChar {
  constructor() {
    this.uchar = " ";
    this.penState = new PenState();
  }
  reset() {
    this.uchar = " ";
    this.penState.reset();
  }
  setChar(uchar, newPenState) {
    this.uchar = uchar;
    this.penState.copy(newPenState);
  }
  setPenState(newPenState) {
    this.penState.copy(newPenState);
  }
  equals(other) {
    return this.uchar === other.uchar && this.penState.equals(other.penState);
  }
  copy(newChar) {
    this.uchar = newChar.uchar;
    this.penState.copy(newChar.penState);
  }
  isEmpty() {
    return this.uchar === " " && this.penState.isDefault();
  }
}
class Row {
  constructor(logger2) {
    this.chars = [];
    this.pos = 0;
    this.currPenState = new PenState();
    this.cueStartTime = null;
    this.logger = void 0;
    for (let i2 = 0; i2 < NR_COLS; i2++) {
      this.chars.push(new StyledUnicodeChar());
    }
    this.logger = logger2;
  }
  equals(other) {
    for (let i2 = 0; i2 < NR_COLS; i2++) {
      if (!this.chars[i2].equals(other.chars[i2])) {
        return false;
      }
    }
    return true;
  }
  copy(other) {
    for (let i2 = 0; i2 < NR_COLS; i2++) {
      this.chars[i2].copy(other.chars[i2]);
    }
  }
  isEmpty() {
    let empty = true;
    for (let i2 = 0; i2 < NR_COLS; i2++) {
      if (!this.chars[i2].isEmpty()) {
        empty = false;
        break;
      }
    }
    return empty;
  }
  /**
   *  Set the cursor to a valid column.
   */
  setCursor(absPos) {
    if (this.pos !== absPos) {
      this.pos = absPos;
    }
    if (this.pos < 0) {
      this.logger.log(3, "Negative cursor position " + this.pos);
      this.pos = 0;
    } else if (this.pos > NR_COLS) {
      this.logger.log(3, "Too large cursor position " + this.pos);
      this.pos = NR_COLS;
    }
  }
  /**
   * Move the cursor relative to current position.
   */
  moveCursor(relPos) {
    const newPos = this.pos + relPos;
    if (relPos > 1) {
      for (let i2 = this.pos + 1; i2 < newPos + 1; i2++) {
        this.chars[i2].setPenState(this.currPenState);
      }
    }
    this.setCursor(newPos);
  }
  /**
   * Backspace, move one step back and clear character.
   */
  backSpace() {
    this.moveCursor(-1);
    this.chars[this.pos].setChar(" ", this.currPenState);
  }
  insertChar(byte) {
    if (byte >= 144) {
      this.backSpace();
    }
    const char = getCharForByte(byte);
    if (this.pos >= NR_COLS) {
      this.logger.log(0, () => "Cannot insert " + byte.toString(16) + " (" + char + ") at position " + this.pos + ". Skipping it!");
      return;
    }
    this.chars[this.pos].setChar(char, this.currPenState);
    this.moveCursor(1);
  }
  clearFromPos(startPos) {
    let i2;
    for (i2 = startPos; i2 < NR_COLS; i2++) {
      this.chars[i2].reset();
    }
  }
  clear() {
    this.clearFromPos(0);
    this.pos = 0;
    this.currPenState.reset();
  }
  clearToEndOfRow() {
    this.clearFromPos(this.pos);
  }
  getTextString() {
    const chars = [];
    let empty = true;
    for (let i2 = 0; i2 < NR_COLS; i2++) {
      const char = this.chars[i2].uchar;
      if (char !== " ") {
        empty = false;
      }
      chars.push(char);
    }
    if (empty) {
      return "";
    } else {
      return chars.join("");
    }
  }
  setPenStyles(styles) {
    this.currPenState.setStyles(styles);
    const currChar = this.chars[this.pos];
    currChar.setPenState(this.currPenState);
  }
}
class CaptionScreen {
  constructor(logger2) {
    this.rows = [];
    this.currRow = NR_ROWS - 1;
    this.nrRollUpRows = null;
    this.lastOutputScreen = null;
    this.logger = void 0;
    for (let i2 = 0; i2 < NR_ROWS; i2++) {
      this.rows.push(new Row(logger2));
    }
    this.logger = logger2;
  }
  reset() {
    for (let i2 = 0; i2 < NR_ROWS; i2++) {
      this.rows[i2].clear();
    }
    this.currRow = NR_ROWS - 1;
  }
  equals(other) {
    let equal = true;
    for (let i2 = 0; i2 < NR_ROWS; i2++) {
      if (!this.rows[i2].equals(other.rows[i2])) {
        equal = false;
        break;
      }
    }
    return equal;
  }
  copy(other) {
    for (let i2 = 0; i2 < NR_ROWS; i2++) {
      this.rows[i2].copy(other.rows[i2]);
    }
  }
  isEmpty() {
    let empty = true;
    for (let i2 = 0; i2 < NR_ROWS; i2++) {
      if (!this.rows[i2].isEmpty()) {
        empty = false;
        break;
      }
    }
    return empty;
  }
  backSpace() {
    const row = this.rows[this.currRow];
    row.backSpace();
  }
  clearToEndOfRow() {
    const row = this.rows[this.currRow];
    row.clearToEndOfRow();
  }
  /**
   * Insert a character (without styling) in the current row.
   */
  insertChar(char) {
    const row = this.rows[this.currRow];
    row.insertChar(char);
  }
  setPen(styles) {
    const row = this.rows[this.currRow];
    row.setPenStyles(styles);
  }
  moveCursor(relPos) {
    const row = this.rows[this.currRow];
    row.moveCursor(relPos);
  }
  setCursor(absPos) {
    this.logger.log(2, "setCursor: " + absPos);
    const row = this.rows[this.currRow];
    row.setCursor(absPos);
  }
  setPAC(pacData) {
    this.logger.log(2, () => "pacData = " + JSON.stringify(pacData));
    let newRow = pacData.row - 1;
    if (this.nrRollUpRows && newRow < this.nrRollUpRows - 1) {
      newRow = this.nrRollUpRows - 1;
    }
    if (this.nrRollUpRows && this.currRow !== newRow) {
      for (let i2 = 0; i2 < NR_ROWS; i2++) {
        this.rows[i2].clear();
      }
      const topRowIndex = this.currRow + 1 - this.nrRollUpRows;
      const lastOutputScreen = this.lastOutputScreen;
      if (lastOutputScreen) {
        const prevLineTime = lastOutputScreen.rows[topRowIndex].cueStartTime;
        const time = this.logger.time;
        if (prevLineTime !== null && time !== null && prevLineTime < time) {
          for (let i2 = 0; i2 < this.nrRollUpRows; i2++) {
            this.rows[newRow - this.nrRollUpRows + i2 + 1].copy(lastOutputScreen.rows[topRowIndex + i2]);
          }
        }
      }
    }
    this.currRow = newRow;
    const row = this.rows[this.currRow];
    if (pacData.indent !== null) {
      const indent = pacData.indent;
      const prevPos = Math.max(indent - 1, 0);
      row.setCursor(pacData.indent);
      pacData.color = row.chars[prevPos].penState.foreground;
    }
    const styles = {
      foreground: pacData.color,
      underline: pacData.underline,
      italics: pacData.italics,
      background: "black",
      flash: false
    };
    this.setPen(styles);
  }
  /**
   * Set background/extra foreground, but first do back_space, and then insert space (backwards compatibility).
   */
  setBkgData(bkgData) {
    this.logger.log(2, () => "bkgData = " + JSON.stringify(bkgData));
    this.backSpace();
    this.setPen(bkgData);
    this.insertChar(32);
  }
  setRollUpRows(nrRows) {
    this.nrRollUpRows = nrRows;
  }
  rollUp() {
    if (this.nrRollUpRows === null) {
      this.logger.log(3, "roll_up but nrRollUpRows not set yet");
      return;
    }
    this.logger.log(1, () => this.getDisplayText());
    const topRowIndex = this.currRow + 1 - this.nrRollUpRows;
    const topRow = this.rows.splice(topRowIndex, 1)[0];
    topRow.clear();
    this.rows.splice(this.currRow, 0, topRow);
    this.logger.log(2, "Rolling up");
  }
  /**
   * Get all non-empty rows with as unicode text.
   */
  getDisplayText(asOneRow) {
    asOneRow = asOneRow || false;
    const displayText = [];
    let text = "";
    let rowNr = -1;
    for (let i2 = 0; i2 < NR_ROWS; i2++) {
      const rowText = this.rows[i2].getTextString();
      if (rowText) {
        rowNr = i2 + 1;
        if (asOneRow) {
          displayText.push("Row " + rowNr + ": '" + rowText + "'");
        } else {
          displayText.push(rowText.trim());
        }
      }
    }
    if (displayText.length > 0) {
      if (asOneRow) {
        text = "[" + displayText.join(" | ") + "]";
      } else {
        text = displayText.join("\n");
      }
    }
    return text;
  }
  getTextAndFormat() {
    return this.rows;
  }
}
class Cea608Channel {
  constructor(channelNumber, outputFilter, logger2) {
    this.chNr = void 0;
    this.outputFilter = void 0;
    this.mode = void 0;
    this.verbose = void 0;
    this.displayedMemory = void 0;
    this.nonDisplayedMemory = void 0;
    this.lastOutputScreen = void 0;
    this.currRollUpRow = void 0;
    this.writeScreen = void 0;
    this.cueStartTime = void 0;
    this.logger = void 0;
    this.chNr = channelNumber;
    this.outputFilter = outputFilter;
    this.mode = null;
    this.verbose = 0;
    this.displayedMemory = new CaptionScreen(logger2);
    this.nonDisplayedMemory = new CaptionScreen(logger2);
    this.lastOutputScreen = new CaptionScreen(logger2);
    this.currRollUpRow = this.displayedMemory.rows[NR_ROWS - 1];
    this.writeScreen = this.displayedMemory;
    this.mode = null;
    this.cueStartTime = null;
    this.logger = logger2;
  }
  reset() {
    this.mode = null;
    this.displayedMemory.reset();
    this.nonDisplayedMemory.reset();
    this.lastOutputScreen.reset();
    this.outputFilter.reset();
    this.currRollUpRow = this.displayedMemory.rows[NR_ROWS - 1];
    this.writeScreen = this.displayedMemory;
    this.mode = null;
    this.cueStartTime = null;
  }
  getHandler() {
    return this.outputFilter;
  }
  setHandler(newHandler) {
    this.outputFilter = newHandler;
  }
  setPAC(pacData) {
    this.writeScreen.setPAC(pacData);
  }
  setBkgData(bkgData) {
    this.writeScreen.setBkgData(bkgData);
  }
  setMode(newMode) {
    if (newMode === this.mode) {
      return;
    }
    this.mode = newMode;
    this.logger.log(2, () => "MODE=" + newMode);
    if (this.mode === "MODE_POP-ON") {
      this.writeScreen = this.nonDisplayedMemory;
    } else {
      this.writeScreen = this.displayedMemory;
      this.writeScreen.reset();
    }
    if (this.mode !== "MODE_ROLL-UP") {
      this.displayedMemory.nrRollUpRows = null;
      this.nonDisplayedMemory.nrRollUpRows = null;
    }
    this.mode = newMode;
  }
  insertChars(chars) {
    for (let i2 = 0; i2 < chars.length; i2++) {
      this.writeScreen.insertChar(chars[i2]);
    }
    const screen = this.writeScreen === this.displayedMemory ? "DISP" : "NON_DISP";
    this.logger.log(2, () => screen + ": " + this.writeScreen.getDisplayText(true));
    if (this.mode === "MODE_PAINT-ON" || this.mode === "MODE_ROLL-UP") {
      this.logger.log(1, () => "DISPLAYED: " + this.displayedMemory.getDisplayText(true));
      this.outputDataUpdate();
    }
  }
  ccRCL() {
    this.logger.log(2, "RCL - Resume Caption Loading");
    this.setMode("MODE_POP-ON");
  }
  ccBS() {
    this.logger.log(2, "BS - BackSpace");
    if (this.mode === "MODE_TEXT") {
      return;
    }
    this.writeScreen.backSpace();
    if (this.writeScreen === this.displayedMemory) {
      this.outputDataUpdate();
    }
  }
  ccAOF() {
  }
  ccAON() {
  }
  ccDER() {
    this.logger.log(2, "DER- Delete to End of Row");
    this.writeScreen.clearToEndOfRow();
    this.outputDataUpdate();
  }
  ccRU(nrRows) {
    this.logger.log(2, "RU(" + nrRows + ") - Roll Up");
    this.writeScreen = this.displayedMemory;
    this.setMode("MODE_ROLL-UP");
    this.writeScreen.setRollUpRows(nrRows);
  }
  ccFON() {
    this.logger.log(2, "FON - Flash On");
    this.writeScreen.setPen({
      flash: true
    });
  }
  ccRDC() {
    this.logger.log(2, "RDC - Resume Direct Captioning");
    this.setMode("MODE_PAINT-ON");
  }
  ccTR() {
    this.logger.log(2, "TR");
    this.setMode("MODE_TEXT");
  }
  ccRTD() {
    this.logger.log(2, "RTD");
    this.setMode("MODE_TEXT");
  }
  ccEDM() {
    this.logger.log(2, "EDM - Erase Displayed Memory");
    this.displayedMemory.reset();
    this.outputDataUpdate(true);
  }
  ccCR() {
    this.logger.log(2, "CR - Carriage Return");
    this.writeScreen.rollUp();
    this.outputDataUpdate(true);
  }
  ccENM() {
    this.logger.log(2, "ENM - Erase Non-displayed Memory");
    this.nonDisplayedMemory.reset();
  }
  ccEOC() {
    this.logger.log(2, "EOC - End Of Caption");
    if (this.mode === "MODE_POP-ON") {
      const tmp = this.displayedMemory;
      this.displayedMemory = this.nonDisplayedMemory;
      this.nonDisplayedMemory = tmp;
      this.writeScreen = this.nonDisplayedMemory;
      this.logger.log(1, () => "DISP: " + this.displayedMemory.getDisplayText());
    }
    this.outputDataUpdate(true);
  }
  ccTO(nrCols) {
    this.logger.log(2, "TO(" + nrCols + ") - Tab Offset");
    this.writeScreen.moveCursor(nrCols);
  }
  ccMIDROW(secondByte) {
    const styles = {
      flash: false
    };
    styles.underline = secondByte % 2 === 1;
    styles.italics = secondByte >= 46;
    if (!styles.italics) {
      const colorIndex = Math.floor(secondByte / 2) - 16;
      const colors = ["white", "green", "blue", "cyan", "red", "yellow", "magenta"];
      styles.foreground = colors[colorIndex];
    } else {
      styles.foreground = "white";
    }
    this.logger.log(2, "MIDROW: " + JSON.stringify(styles));
    this.writeScreen.setPen(styles);
  }
  outputDataUpdate(dispatch = false) {
    const time = this.logger.time;
    if (time === null) {
      return;
    }
    if (this.outputFilter) {
      if (this.cueStartTime === null && !this.displayedMemory.isEmpty()) {
        this.cueStartTime = time;
      } else {
        if (!this.displayedMemory.equals(this.lastOutputScreen)) {
          this.outputFilter.newCue(this.cueStartTime, time, this.lastOutputScreen);
          if (dispatch && this.outputFilter.dispatchCue) {
            this.outputFilter.dispatchCue();
          }
          this.cueStartTime = this.displayedMemory.isEmpty() ? null : time;
        }
      }
      this.lastOutputScreen.copy(this.displayedMemory);
    }
  }
  cueSplitAtTime(t2) {
    if (this.outputFilter) {
      if (!this.displayedMemory.isEmpty()) {
        if (this.outputFilter.newCue) {
          this.outputFilter.newCue(this.cueStartTime, t2, this.displayedMemory);
        }
        this.cueStartTime = t2;
      }
    }
  }
}
class Cea608Parser {
  constructor(field, out1, out2) {
    this.channels = void 0;
    this.currentChannel = 0;
    this.cmdHistory = createCmdHistory();
    this.logger = void 0;
    const logger2 = this.logger = new CaptionsLogger();
    this.channels = [null, new Cea608Channel(field, out1, logger2), new Cea608Channel(field + 1, out2, logger2)];
  }
  getHandler(channel) {
    return this.channels[channel].getHandler();
  }
  setHandler(channel, newHandler) {
    this.channels[channel].setHandler(newHandler);
  }
  /**
   * Add data for time t in forms of list of bytes (unsigned ints). The bytes are treated as pairs.
   */
  addData(time, byteList) {
    this.logger.time = time;
    for (let i2 = 0; i2 < byteList.length; i2 += 2) {
      const a = byteList[i2] & 127;
      const b = byteList[i2 + 1] & 127;
      let cmdFound = false;
      let charsFound = null;
      if (a === 0 && b === 0) {
        continue;
      } else {
        this.logger.log(3, () => "[" + numArrayToHexArray([byteList[i2], byteList[i2 + 1]]) + "] -> (" + numArrayToHexArray([a, b]) + ")");
      }
      const cmdHistory = this.cmdHistory;
      const isControlCode = a >= 16 && a <= 31;
      if (isControlCode) {
        if (hasCmdRepeated(a, b, cmdHistory)) {
          setLastCmd(null, null, cmdHistory);
          this.logger.log(3, () => "Repeated command (" + numArrayToHexArray([a, b]) + ") is dropped");
          continue;
        }
        setLastCmd(a, b, this.cmdHistory);
        cmdFound = this.parseCmd(a, b);
        if (!cmdFound) {
          cmdFound = this.parseMidrow(a, b);
        }
        if (!cmdFound) {
          cmdFound = this.parsePAC(a, b);
        }
        if (!cmdFound) {
          cmdFound = this.parseBackgroundAttributes(a, b);
        }
      } else {
        setLastCmd(null, null, cmdHistory);
      }
      if (!cmdFound) {
        charsFound = this.parseChars(a, b);
        if (charsFound) {
          const currChNr = this.currentChannel;
          if (currChNr && currChNr > 0) {
            const channel = this.channels[currChNr];
            channel.insertChars(charsFound);
          } else {
            this.logger.log(2, "No channel found yet. TEXT-MODE?");
          }
        }
      }
      if (!cmdFound && !charsFound) {
        this.logger.log(2, () => "Couldn't parse cleaned data " + numArrayToHexArray([a, b]) + " orig: " + numArrayToHexArray([byteList[i2], byteList[i2 + 1]]));
      }
    }
  }
  /**
   * Parse Command.
   * @returns True if a command was found
   */
  parseCmd(a, b) {
    const cond1 = (a === 20 || a === 28 || a === 21 || a === 29) && b >= 32 && b <= 47;
    const cond2 = (a === 23 || a === 31) && b >= 33 && b <= 35;
    if (!(cond1 || cond2)) {
      return false;
    }
    const chNr = a === 20 || a === 21 || a === 23 ? 1 : 2;
    const channel = this.channels[chNr];
    if (a === 20 || a === 21 || a === 28 || a === 29) {
      if (b === 32) {
        channel.ccRCL();
      } else if (b === 33) {
        channel.ccBS();
      } else if (b === 34) {
        channel.ccAOF();
      } else if (b === 35) {
        channel.ccAON();
      } else if (b === 36) {
        channel.ccDER();
      } else if (b === 37) {
        channel.ccRU(2);
      } else if (b === 38) {
        channel.ccRU(3);
      } else if (b === 39) {
        channel.ccRU(4);
      } else if (b === 40) {
        channel.ccFON();
      } else if (b === 41) {
        channel.ccRDC();
      } else if (b === 42) {
        channel.ccTR();
      } else if (b === 43) {
        channel.ccRTD();
      } else if (b === 44) {
        channel.ccEDM();
      } else if (b === 45) {
        channel.ccCR();
      } else if (b === 46) {
        channel.ccENM();
      } else if (b === 47) {
        channel.ccEOC();
      }
    } else {
      channel.ccTO(b - 32);
    }
    this.currentChannel = chNr;
    return true;
  }
  /**
   * Parse midrow styling command
   */
  parseMidrow(a, b) {
    let chNr = 0;
    if ((a === 17 || a === 25) && b >= 32 && b <= 47) {
      if (a === 17) {
        chNr = 1;
      } else {
        chNr = 2;
      }
      if (chNr !== this.currentChannel) {
        this.logger.log(0, "Mismatch channel in midrow parsing");
        return false;
      }
      const channel = this.channels[chNr];
      if (!channel) {
        return false;
      }
      channel.ccMIDROW(b);
      this.logger.log(3, () => "MIDROW (" + numArrayToHexArray([a, b]) + ")");
      return true;
    }
    return false;
  }
  /**
   * Parse Preable Access Codes (Table 53).
   * @returns {Boolean} Tells if PAC found
   */
  parsePAC(a, b) {
    let row;
    const case1 = (a >= 17 && a <= 23 || a >= 25 && a <= 31) && b >= 64 && b <= 127;
    const case2 = (a === 16 || a === 24) && b >= 64 && b <= 95;
    if (!(case1 || case2)) {
      return false;
    }
    const chNr = a <= 23 ? 1 : 2;
    if (b >= 64 && b <= 95) {
      row = chNr === 1 ? rowsLowCh1[a] : rowsLowCh2[a];
    } else {
      row = chNr === 1 ? rowsHighCh1[a] : rowsHighCh2[a];
    }
    const channel = this.channels[chNr];
    if (!channel) {
      return false;
    }
    channel.setPAC(this.interpretPAC(row, b));
    this.currentChannel = chNr;
    return true;
  }
  /**
   * Interpret the second byte of the pac, and return the information.
   * @returns pacData with style parameters
   */
  interpretPAC(row, byte) {
    let pacIndex;
    const pacData = {
      color: null,
      italics: false,
      indent: null,
      underline: false,
      row
    };
    if (byte > 95) {
      pacIndex = byte - 96;
    } else {
      pacIndex = byte - 64;
    }
    pacData.underline = (pacIndex & 1) === 1;
    if (pacIndex <= 13) {
      pacData.color = ["white", "green", "blue", "cyan", "red", "yellow", "magenta", "white"][Math.floor(pacIndex / 2)];
    } else if (pacIndex <= 15) {
      pacData.italics = true;
      pacData.color = "white";
    } else {
      pacData.indent = Math.floor((pacIndex - 16) / 2) * 4;
    }
    return pacData;
  }
  /**
   * Parse characters.
   * @returns An array with 1 to 2 codes corresponding to chars, if found. null otherwise.
   */
  parseChars(a, b) {
    let channelNr;
    let charCodes = null;
    let charCode1 = null;
    if (a >= 25) {
      channelNr = 2;
      charCode1 = a - 8;
    } else {
      channelNr = 1;
      charCode1 = a;
    }
    if (charCode1 >= 17 && charCode1 <= 19) {
      let oneCode;
      if (charCode1 === 17) {
        oneCode = b + 80;
      } else if (charCode1 === 18) {
        oneCode = b + 112;
      } else {
        oneCode = b + 144;
      }
      this.logger.log(2, () => "Special char '" + getCharForByte(oneCode) + "' in channel " + channelNr);
      charCodes = [oneCode];
    } else if (a >= 32 && a <= 127) {
      charCodes = b === 0 ? [a] : [a, b];
    }
    if (charCodes) {
      this.logger.log(3, () => "Char codes =  " + numArrayToHexArray(charCodes).join(","));
    }
    return charCodes;
  }
  /**
   * Parse extended background attributes as well as new foreground color black.
   * @returns True if background attributes are found
   */
  parseBackgroundAttributes(a, b) {
    const case1 = (a === 16 || a === 24) && b >= 32 && b <= 47;
    const case2 = (a === 23 || a === 31) && b >= 45 && b <= 47;
    if (!(case1 || case2)) {
      return false;
    }
    let index;
    const bkgData = {};
    if (a === 16 || a === 24) {
      index = Math.floor((b - 32) / 2);
      bkgData.background = backgroundColors[index];
      if (b % 2 === 1) {
        bkgData.background = bkgData.background + "_semi";
      }
    } else if (b === 45) {
      bkgData.background = "transparent";
    } else {
      bkgData.foreground = "black";
      if (b === 47) {
        bkgData.underline = true;
      }
    }
    const chNr = a <= 23 ? 1 : 2;
    const channel = this.channels[chNr];
    channel.setBkgData(bkgData);
    return true;
  }
  /**
   * Reset state of parser and its channels.
   */
  reset() {
    for (let i2 = 0; i2 < Object.keys(this.channels).length; i2++) {
      const channel = this.channels[i2];
      if (channel) {
        channel.reset();
      }
    }
    setLastCmd(null, null, this.cmdHistory);
  }
  /**
   * Trigger the generation of a cue, and the start of a new one if displayScreens are not empty.
   */
  cueSplitAtTime(t2) {
    for (let i2 = 0; i2 < this.channels.length; i2++) {
      const channel = this.channels[i2];
      if (channel) {
        channel.cueSplitAtTime(t2);
      }
    }
  }
}
function setLastCmd(a, b, cmdHistory) {
  cmdHistory.a = a;
  cmdHistory.b = b;
}
function hasCmdRepeated(a, b, cmdHistory) {
  return cmdHistory.a === a && cmdHistory.b === b;
}
function createCmdHistory() {
  return {
    a: null,
    b: null
  };
}
class OutputFilter {
  constructor(timelineController, trackName) {
    this.timelineController = void 0;
    this.cueRanges = [];
    this.trackName = void 0;
    this.startTime = null;
    this.endTime = null;
    this.screen = null;
    this.timelineController = timelineController;
    this.trackName = trackName;
  }
  dispatchCue() {
    if (this.startTime === null) {
      return;
    }
    this.timelineController.addCues(this.trackName, this.startTime, this.endTime, this.screen, this.cueRanges);
    this.startTime = null;
  }
  newCue(startTime, endTime, screen) {
    if (this.startTime === null || this.startTime > startTime) {
      this.startTime = startTime;
    }
    this.endTime = endTime;
    this.screen = screen;
    this.timelineController.createCaptionsTrack(this.trackName);
  }
  reset() {
    this.cueRanges = [];
    this.startTime = null;
  }
}
var VTTCue = (function() {
  if (optionalSelf != null && optionalSelf.VTTCue) {
    return self.VTTCue;
  }
  const AllowedDirections = ["", "lr", "rl"];
  const AllowedAlignments = ["start", "middle", "end", "left", "right"];
  function isAllowedValue(allowed, value) {
    if (typeof value !== "string") {
      return false;
    }
    if (!Array.isArray(allowed)) {
      return false;
    }
    const lcValue = value.toLowerCase();
    if (~allowed.indexOf(lcValue)) {
      return lcValue;
    }
    return false;
  }
  function findDirectionSetting(value) {
    return isAllowedValue(AllowedDirections, value);
  }
  function findAlignSetting(value) {
    return isAllowedValue(AllowedAlignments, value);
  }
  function extend(obj, ...rest) {
    let i2 = 1;
    for (; i2 < arguments.length; i2++) {
      const cobj = arguments[i2];
      for (const p in cobj) {
        obj[p] = cobj[p];
      }
    }
    return obj;
  }
  function VTTCue2(startTime, endTime, text) {
    const cue = this;
    const baseObj = {
      enumerable: true
    };
    cue.hasBeenReset = false;
    let _id = "";
    let _pauseOnExit = false;
    let _startTime = startTime;
    let _endTime = endTime;
    let _text = text;
    let _region = null;
    let _vertical = "";
    let _snapToLines = true;
    let _line = "auto";
    let _lineAlign = "start";
    let _position = 50;
    let _positionAlign = "middle";
    let _size = 50;
    let _align = "middle";
    Object.defineProperty(cue, "id", extend({}, baseObj, {
      get: function() {
        return _id;
      },
      set: function(value) {
        _id = "" + value;
      }
    }));
    Object.defineProperty(cue, "pauseOnExit", extend({}, baseObj, {
      get: function() {
        return _pauseOnExit;
      },
      set: function(value) {
        _pauseOnExit = !!value;
      }
    }));
    Object.defineProperty(cue, "startTime", extend({}, baseObj, {
      get: function() {
        return _startTime;
      },
      set: function(value) {
        if (typeof value !== "number") {
          throw new TypeError("Start time must be set to a number.");
        }
        _startTime = value;
        this.hasBeenReset = true;
      }
    }));
    Object.defineProperty(cue, "endTime", extend({}, baseObj, {
      get: function() {
        return _endTime;
      },
      set: function(value) {
        if (typeof value !== "number") {
          throw new TypeError("End time must be set to a number.");
        }
        _endTime = value;
        this.hasBeenReset = true;
      }
    }));
    Object.defineProperty(cue, "text", extend({}, baseObj, {
      get: function() {
        return _text;
      },
      set: function(value) {
        _text = "" + value;
        this.hasBeenReset = true;
      }
    }));
    Object.defineProperty(cue, "region", extend({}, baseObj, {
      get: function() {
        return _region;
      },
      set: function(value) {
        _region = value;
        this.hasBeenReset = true;
      }
    }));
    Object.defineProperty(cue, "vertical", extend({}, baseObj, {
      get: function() {
        return _vertical;
      },
      set: function(value) {
        const setting = findDirectionSetting(value);
        if (setting === false) {
          throw new SyntaxError("An invalid or illegal string was specified.");
        }
        _vertical = setting;
        this.hasBeenReset = true;
      }
    }));
    Object.defineProperty(cue, "snapToLines", extend({}, baseObj, {
      get: function() {
        return _snapToLines;
      },
      set: function(value) {
        _snapToLines = !!value;
        this.hasBeenReset = true;
      }
    }));
    Object.defineProperty(cue, "line", extend({}, baseObj, {
      get: function() {
        return _line;
      },
      set: function(value) {
        if (typeof value !== "number" && value !== "auto") {
          throw new SyntaxError("An invalid number or illegal string was specified.");
        }
        _line = value;
        this.hasBeenReset = true;
      }
    }));
    Object.defineProperty(cue, "lineAlign", extend({}, baseObj, {
      get: function() {
        return _lineAlign;
      },
      set: function(value) {
        const setting = findAlignSetting(value);
        if (!setting) {
          throw new SyntaxError("An invalid or illegal string was specified.");
        }
        _lineAlign = setting;
        this.hasBeenReset = true;
      }
    }));
    Object.defineProperty(cue, "position", extend({}, baseObj, {
      get: function() {
        return _position;
      },
      set: function(value) {
        if (value < 0 || value > 100) {
          throw new Error("Position must be between 0 and 100.");
        }
        _position = value;
        this.hasBeenReset = true;
      }
    }));
    Object.defineProperty(cue, "positionAlign", extend({}, baseObj, {
      get: function() {
        return _positionAlign;
      },
      set: function(value) {
        const setting = findAlignSetting(value);
        if (!setting) {
          throw new SyntaxError("An invalid or illegal string was specified.");
        }
        _positionAlign = setting;
        this.hasBeenReset = true;
      }
    }));
    Object.defineProperty(cue, "size", extend({}, baseObj, {
      get: function() {
        return _size;
      },
      set: function(value) {
        if (value < 0 || value > 100) {
          throw new Error("Size must be between 0 and 100.");
        }
        _size = value;
        this.hasBeenReset = true;
      }
    }));
    Object.defineProperty(cue, "align", extend({}, baseObj, {
      get: function() {
        return _align;
      },
      set: function(value) {
        const setting = findAlignSetting(value);
        if (!setting) {
          throw new SyntaxError("An invalid or illegal string was specified.");
        }
        _align = setting;
        this.hasBeenReset = true;
      }
    }));
    cue.displayState = void 0;
  }
  VTTCue2.prototype.getCueAsHTML = function() {
    const WebVTT = self.WebVTT;
    return WebVTT.convertCueToDOMTree(self, this.text);
  };
  return VTTCue2;
})();
class StringDecoder {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  decode(data, options) {
    if (!data) {
      return "";
    }
    if (typeof data !== "string") {
      throw new Error("Error - expected string data.");
    }
    return decodeURIComponent(encodeURIComponent(data));
  }
}
function parseTimeStamp(input) {
  function computeSeconds(h, m2, s2, f) {
    return (h | 0) * 3600 + (m2 | 0) * 60 + (s2 | 0) + parseFloat(f || 0);
  }
  const m = input.match(/^(?:(\d+):)?(\d{2}):(\d{2})(\.\d+)?/);
  if (!m) {
    return null;
  }
  if (parseFloat(m[2]) > 59) {
    return computeSeconds(m[2], m[3], 0, m[4]);
  }
  return computeSeconds(m[1], m[2], m[3], m[4]);
}
class Settings {
  constructor() {
    this.values = /* @__PURE__ */ Object.create(null);
  }
  // Only accept the first assignment to any key.
  set(k, v) {
    if (!this.get(k) && v !== "") {
      this.values[k] = v;
    }
  }
  // Return the value for a key, or a default value.
  // If 'defaultKey' is passed then 'dflt' is assumed to be an object with
  // a number of possible default values as properties where 'defaultKey' is
  // the key of the property that will be chosen; otherwise it's assumed to be
  // a single value.
  get(k, dflt, defaultKey) {
    if (defaultKey) {
      return this.has(k) ? this.values[k] : dflt[defaultKey];
    }
    return this.has(k) ? this.values[k] : dflt;
  }
  // Check whether we have a value for a key.
  has(k) {
    return k in this.values;
  }
  // Accept a setting if its one of the given alternatives.
  alt(k, v, a) {
    for (let n2 = 0; n2 < a.length; ++n2) {
      if (v === a[n2]) {
        this.set(k, v);
        break;
      }
    }
  }
  // Accept a setting if its a valid (signed) integer.
  integer(k, v) {
    if (/^-?\d+$/.test(v)) {
      this.set(k, parseInt(v, 10));
    }
  }
  // Accept a setting if its a valid percentage.
  percent(k, v) {
    if (/^([\d]{1,3})(\.[\d]*)?%$/.test(v)) {
      const percent = parseFloat(v);
      if (percent >= 0 && percent <= 100) {
        this.set(k, percent);
        return true;
      }
    }
    return false;
  }
}
function parseOptions(input, callback, keyValueDelim, groupDelim) {
  const groups = groupDelim ? input.split(groupDelim) : [input];
  for (const i2 in groups) {
    if (typeof groups[i2] !== "string") {
      continue;
    }
    const kv = groups[i2].split(keyValueDelim);
    if (kv.length !== 2) {
      continue;
    }
    const k = kv[0];
    const v = kv[1];
    callback(k, v);
  }
}
const defaults = new VTTCue(0, 0, "");
const center = defaults.align === "middle" ? "middle" : "center";
function parseCue(input, cue, regionList) {
  const oInput = input;
  function consumeTimeStamp() {
    const ts = parseTimeStamp(input);
    if (ts === null) {
      throw new Error("Malformed timestamp: " + oInput);
    }
    input = input.replace(/^[^\sa-zA-Z-]+/, "");
    return ts;
  }
  function consumeCueSettings(input2, cue2) {
    const settings = new Settings();
    parseOptions(input2, function(k, v) {
      let vals;
      switch (k) {
        case "region":
          for (let i2 = regionList.length - 1; i2 >= 0; i2--) {
            if (regionList[i2].id === v) {
              settings.set(k, regionList[i2].region);
              break;
            }
          }
          break;
        case "vertical":
          settings.alt(k, v, ["rl", "lr"]);
          break;
        case "line":
          vals = v.split(",");
          settings.integer(k, vals[0]);
          if (settings.percent(k, vals[0])) {
            settings.set("snapToLines", false);
          }
          settings.alt(k, vals[0], ["auto"]);
          if (vals.length === 2) {
            settings.alt("lineAlign", vals[1], ["start", center, "end"]);
          }
          break;
        case "position":
          vals = v.split(",");
          settings.percent(k, vals[0]);
          if (vals.length === 2) {
            settings.alt("positionAlign", vals[1], ["start", center, "end", "line-left", "line-right", "auto"]);
          }
          break;
        case "size":
          settings.percent(k, v);
          break;
        case "align":
          settings.alt(k, v, ["start", center, "end", "left", "right"]);
          break;
      }
    }, /:/, /\s/);
    cue2.region = settings.get("region", null);
    cue2.vertical = settings.get("vertical", "");
    let line = settings.get("line", "auto");
    if (line === "auto" && defaults.line === -1) {
      line = -1;
    }
    cue2.line = line;
    cue2.lineAlign = settings.get("lineAlign", "start");
    cue2.snapToLines = settings.get("snapToLines", true);
    cue2.size = settings.get("size", 100);
    cue2.align = settings.get("align", center);
    let position = settings.get("position", "auto");
    if (position === "auto" && defaults.position === 50) {
      position = cue2.align === "start" || cue2.align === "left" ? 0 : cue2.align === "end" || cue2.align === "right" ? 100 : 50;
    }
    cue2.position = position;
  }
  function skipWhitespace() {
    input = input.replace(/^\s+/, "");
  }
  skipWhitespace();
  cue.startTime = consumeTimeStamp();
  skipWhitespace();
  if (input.slice(0, 3) !== "-->") {
    throw new Error("Malformed time stamp (time stamps must be separated by '-->'): " + oInput);
  }
  input = input.slice(3);
  skipWhitespace();
  cue.endTime = consumeTimeStamp();
  skipWhitespace();
  consumeCueSettings(input, cue);
}
function fixLineBreaks(input) {
  return input.replace(/<br(?: \/)?>/gi, "\n");
}
class VTTParser {
  constructor() {
    this.state = "INITIAL";
    this.buffer = "";
    this.decoder = new StringDecoder();
    this.regionList = [];
    this.cue = null;
    this.oncue = void 0;
    this.onparsingerror = void 0;
    this.onflush = void 0;
  }
  parse(data) {
    const _this = this;
    if (data) {
      _this.buffer += _this.decoder.decode(data, {
        stream: true
      });
    }
    function collectNextLine() {
      let buffer = _this.buffer;
      let pos = 0;
      buffer = fixLineBreaks(buffer);
      while (pos < buffer.length && buffer[pos] !== "\r" && buffer[pos] !== "\n") {
        ++pos;
      }
      const line = buffer.slice(0, pos);
      if (buffer[pos] === "\r") {
        ++pos;
      }
      if (buffer[pos] === "\n") {
        ++pos;
      }
      _this.buffer = buffer.slice(pos);
      return line;
    }
    function parseHeader2(input) {
      parseOptions(input, function(k, v) {
      }, /:/);
    }
    try {
      let line = "";
      if (_this.state === "INITIAL") {
        if (!/\r\n|\n/.test(_this.buffer)) {
          return this;
        }
        line = collectNextLine();
        const m = line.match(/^(ï»¿)?WEBVTT([ \t].*)?$/);
        if (!(m != null && m[0])) {
          throw new Error("Malformed WebVTT signature.");
        }
        _this.state = "HEADER";
      }
      let alreadyCollectedLine = false;
      while (_this.buffer) {
        if (!/\r\n|\n/.test(_this.buffer)) {
          return this;
        }
        if (!alreadyCollectedLine) {
          line = collectNextLine();
        } else {
          alreadyCollectedLine = false;
        }
        switch (_this.state) {
          case "HEADER":
            if (/:/.test(line)) {
              parseHeader2(line);
            } else if (!line) {
              _this.state = "ID";
            }
            continue;
          case "NOTE":
            if (!line) {
              _this.state = "ID";
            }
            continue;
          case "ID":
            if (/^NOTE($|[ \t])/.test(line)) {
              _this.state = "NOTE";
              break;
            }
            if (!line) {
              continue;
            }
            _this.cue = new VTTCue(0, 0, "");
            _this.state = "CUE";
            if (line.indexOf("-->") === -1) {
              _this.cue.id = line;
              continue;
            }
          // Process line as start of a cue.
          /* falls through */
          case "CUE":
            if (!_this.cue) {
              _this.state = "BADCUE";
              continue;
            }
            try {
              parseCue(line, _this.cue, _this.regionList);
            } catch (e2) {
              _this.cue = null;
              _this.state = "BADCUE";
              continue;
            }
            _this.state = "CUETEXT";
            continue;
          case "CUETEXT":
            {
              const hasSubstring = line.indexOf("-->") !== -1;
              if (!line || hasSubstring && (alreadyCollectedLine = true)) {
                if (_this.oncue && _this.cue) {
                  _this.oncue(_this.cue);
                }
                _this.cue = null;
                _this.state = "ID";
                continue;
              }
              if (_this.cue === null) {
                continue;
              }
              if (_this.cue.text) {
                _this.cue.text += "\n";
              }
              _this.cue.text += line;
            }
            continue;
          case "BADCUE":
            if (!line) {
              _this.state = "ID";
            }
        }
      }
    } catch (e2) {
      if (_this.state === "CUETEXT" && _this.cue && _this.oncue) {
        _this.oncue(_this.cue);
      }
      _this.cue = null;
      _this.state = _this.state === "INITIAL" ? "BADWEBVTT" : "BADCUE";
    }
    return this;
  }
  flush() {
    const _this = this;
    try {
      if (_this.cue || _this.state === "HEADER") {
        _this.buffer += "\n\n";
        _this.parse();
      }
      if (_this.state === "INITIAL" || _this.state === "BADWEBVTT") {
        throw new Error("Malformed WebVTT signature.");
      }
    } catch (e2) {
      if (_this.onparsingerror) {
        _this.onparsingerror(e2);
      }
    }
    if (_this.onflush) {
      _this.onflush();
    }
    return this;
  }
}
const LINEBREAKS = /\r\n|\n\r|\n|\r/g;
const startsWith = function startsWith2(inputString, searchString, position = 0) {
  return inputString.slice(position, position + searchString.length) === searchString;
};
const cueString2millis = function cueString2millis2(timeString) {
  let ts = parseInt(timeString.slice(-3));
  const secs = parseInt(timeString.slice(-6, -4));
  const mins = parseInt(timeString.slice(-9, -7));
  const hours = timeString.length > 9 ? parseInt(timeString.substring(0, timeString.indexOf(":"))) : 0;
  if (!isFiniteNumber(ts) || !isFiniteNumber(secs) || !isFiniteNumber(mins) || !isFiniteNumber(hours)) {
    throw Error(`Malformed X-TIMESTAMP-MAP: Local:${timeString}`);
  }
  ts += 1e3 * secs;
  ts += 60 * 1e3 * mins;
  ts += 60 * 60 * 1e3 * hours;
  return ts;
};
const hash = function hash2(text) {
  let _hash = 5381;
  let i2 = text.length;
  while (i2) {
    _hash = _hash * 33 ^ text.charCodeAt(--i2);
  }
  return (_hash >>> 0).toString();
};
function generateCueId(startTime, endTime, text) {
  return hash(startTime.toString()) + hash(endTime.toString()) + hash(text);
}
const calculateOffset = function calculateOffset2(vttCCs, cc, presentationTime) {
  let currCC = vttCCs[cc];
  let prevCC = vttCCs[currCC.prevCC];
  if (!prevCC || !prevCC.new && currCC.new) {
    vttCCs.ccOffset = vttCCs.presentationOffset = currCC.start;
    currCC.new = false;
    return;
  }
  while ((_prevCC = prevCC) != null && _prevCC.new) {
    var _prevCC;
    vttCCs.ccOffset += currCC.start - prevCC.start;
    currCC.new = false;
    currCC = prevCC;
    prevCC = vttCCs[currCC.prevCC];
  }
  vttCCs.presentationOffset = presentationTime;
};
function parseWebVTT(vttByteArray, initPTS, vttCCs, cc, timeOffset, callBack, errorCallBack) {
  const parser = new VTTParser();
  const vttLines = utf8ArrayToStr(new Uint8Array(vttByteArray)).trim().replace(LINEBREAKS, "\n").split("\n");
  const cues = [];
  const init90kHz = initPTS ? toMpegTsClockFromTimescale(initPTS.baseTime, initPTS.timescale) : 0;
  let cueTime = "00:00.000";
  let timestampMapMPEGTS = 0;
  let timestampMapLOCAL = 0;
  let parsingError;
  let inHeader = true;
  parser.oncue = function(cue) {
    const currCC = vttCCs[cc];
    let cueOffset = vttCCs.ccOffset;
    const webVttMpegTsMapOffset = (timestampMapMPEGTS - init90kHz) / 9e4;
    if (currCC != null && currCC.new) {
      if (timestampMapLOCAL !== void 0) {
        cueOffset = vttCCs.ccOffset = currCC.start;
      } else {
        calculateOffset(vttCCs, cc, webVttMpegTsMapOffset);
      }
    }
    if (webVttMpegTsMapOffset) {
      if (!initPTS) {
        parsingError = new Error("Missing initPTS for VTT MPEGTS");
        return;
      }
      cueOffset = webVttMpegTsMapOffset - vttCCs.presentationOffset;
    }
    const duration = cue.endTime - cue.startTime;
    const startTime = normalizePts((cue.startTime + cueOffset - timestampMapLOCAL) * 9e4, timeOffset * 9e4) / 9e4;
    cue.startTime = Math.max(startTime, 0);
    cue.endTime = Math.max(startTime + duration, 0);
    const text = cue.text.trim();
    cue.text = decodeURIComponent(encodeURIComponent(text));
    if (!cue.id) {
      cue.id = generateCueId(cue.startTime, cue.endTime, text);
    }
    if (cue.endTime > 0) {
      cues.push(cue);
    }
  };
  parser.onparsingerror = function(error) {
    parsingError = error;
  };
  parser.onflush = function() {
    if (parsingError) {
      errorCallBack(parsingError);
      return;
    }
    callBack(cues);
  };
  vttLines.forEach((line) => {
    if (inHeader) {
      if (startsWith(line, "X-TIMESTAMP-MAP=")) {
        inHeader = false;
        line.slice(16).split(",").forEach((timestamp) => {
          if (startsWith(timestamp, "LOCAL:")) {
            cueTime = timestamp.slice(6);
          } else if (startsWith(timestamp, "MPEGTS:")) {
            timestampMapMPEGTS = parseInt(timestamp.slice(7));
          }
        });
        try {
          timestampMapLOCAL = cueString2millis(cueTime) / 1e3;
        } catch (error) {
          parsingError = error;
        }
        return;
      } else if (line === "") {
        inHeader = false;
      }
    }
    parser.parse(line + "\n");
  });
  parser.flush();
}
const IMSC1_CODEC = "stpp.ttml.im1t";
const HMSF_REGEX = /^(\d{2,}):(\d{2}):(\d{2}):(\d{2})\.?(\d+)?$/;
const TIME_UNIT_REGEX = /^(\d*(?:\.\d*)?)(h|m|s|ms|f|t)$/;
const textAlignToLineAlign = {
  left: "start",
  center: "center",
  right: "end",
  start: "start",
  end: "end"
};
function parseIMSC1(payload, initPTS, callBack, errorCallBack) {
  const results = findBox(new Uint8Array(payload), ["mdat"]);
  if (results.length === 0) {
    errorCallBack(new Error("Could not parse IMSC1 mdat"));
    return;
  }
  const ttmlList = results.map((mdat) => utf8ArrayToStr(mdat));
  const syncTime = toTimescaleFromScale(initPTS.baseTime, 1, initPTS.timescale);
  try {
    ttmlList.forEach((ttml) => callBack(parseTTML(ttml, syncTime)));
  } catch (error) {
    errorCallBack(error);
  }
}
function parseTTML(ttml, syncTime) {
  const parser = new DOMParser();
  const xmlDoc = parser.parseFromString(ttml, "text/xml");
  const tt = xmlDoc.getElementsByTagName("tt")[0];
  if (!tt) {
    throw new Error("Invalid ttml");
  }
  const defaultRateInfo = {
    frameRate: 30,
    subFrameRate: 1,
    frameRateMultiplier: 0,
    tickRate: 0
  };
  const rateInfo = Object.keys(defaultRateInfo).reduce((result, key) => {
    result[key] = tt.getAttribute(`ttp:${key}`) || defaultRateInfo[key];
    return result;
  }, {});
  const trim = tt.getAttribute("xml:space") !== "preserve";
  const styleElements = collectionToDictionary(getElementCollection(tt, "styling", "style"));
  const regionElements = collectionToDictionary(getElementCollection(tt, "layout", "region"));
  const cueElements = getElementCollection(tt, "body", "[begin]");
  return [].map.call(cueElements, (cueElement) => {
    const cueText = getTextContent(cueElement, trim);
    if (!cueText || !cueElement.hasAttribute("begin")) {
      return null;
    }
    const startTime = parseTtmlTime(cueElement.getAttribute("begin"), rateInfo);
    const duration = parseTtmlTime(cueElement.getAttribute("dur"), rateInfo);
    let endTime = parseTtmlTime(cueElement.getAttribute("end"), rateInfo);
    if (startTime === null) {
      throw timestampParsingError(cueElement);
    }
    if (endTime === null) {
      if (duration === null) {
        throw timestampParsingError(cueElement);
      }
      endTime = startTime + duration;
    }
    const cue = new VTTCue(startTime - syncTime, endTime - syncTime, cueText);
    cue.id = generateCueId(cue.startTime, cue.endTime, cue.text);
    const region = regionElements[cueElement.getAttribute("region")];
    const style = styleElements[cueElement.getAttribute("style")];
    const styles = getTtmlStyles(region, style, styleElements);
    const {
      textAlign
    } = styles;
    if (textAlign) {
      const lineAlign = textAlignToLineAlign[textAlign];
      if (lineAlign) {
        cue.lineAlign = lineAlign;
      }
      cue.align = textAlign;
    }
    _extends(cue, styles);
    return cue;
  }).filter((cue) => cue !== null);
}
function getElementCollection(fromElement, parentName, childName) {
  const parent = fromElement.getElementsByTagName(parentName)[0];
  if (parent) {
    return [].slice.call(parent.querySelectorAll(childName));
  }
  return [];
}
function collectionToDictionary(elementsWithId) {
  return elementsWithId.reduce((dict, element) => {
    const id = element.getAttribute("xml:id");
    if (id) {
      dict[id] = element;
    }
    return dict;
  }, {});
}
function getTextContent(element, trim) {
  return [].slice.call(element.childNodes).reduce((str, node, i2) => {
    var _node$childNodes;
    if (node.nodeName === "br" && i2) {
      return str + "\n";
    }
    if ((_node$childNodes = node.childNodes) != null && _node$childNodes.length) {
      return getTextContent(node, trim);
    } else if (trim) {
      return str + node.textContent.trim().replace(/\s+/g, " ");
    }
    return str + node.textContent;
  }, "");
}
function getTtmlStyles(region, style, styleElements) {
  const ttsNs = "http://www.w3.org/ns/ttml#styling";
  let regionStyle = null;
  const styleAttributes = [
    "displayAlign",
    "textAlign",
    "color",
    "backgroundColor",
    "fontSize",
    "fontFamily"
    // 'fontWeight',
    // 'lineHeight',
    // 'wrapOption',
    // 'fontStyle',
    // 'direction',
    // 'writingMode'
  ];
  const regionStyleName = region != null && region.hasAttribute("style") ? region.getAttribute("style") : null;
  if (regionStyleName && styleElements.hasOwnProperty(regionStyleName)) {
    regionStyle = styleElements[regionStyleName];
  }
  return styleAttributes.reduce((styles, name) => {
    const value = getAttributeNS(style, ttsNs, name) || getAttributeNS(region, ttsNs, name) || getAttributeNS(regionStyle, ttsNs, name);
    if (value) {
      styles[name] = value;
    }
    return styles;
  }, {});
}
function getAttributeNS(element, ns, name) {
  if (!element) {
    return null;
  }
  return element.hasAttributeNS(ns, name) ? element.getAttributeNS(ns, name) : null;
}
function timestampParsingError(node) {
  return new Error(`Could not parse ttml timestamp ${node}`);
}
function parseTtmlTime(timeAttributeValue, rateInfo) {
  if (!timeAttributeValue) {
    return null;
  }
  let seconds = parseTimeStamp(timeAttributeValue);
  if (seconds === null) {
    if (HMSF_REGEX.test(timeAttributeValue)) {
      seconds = parseHoursMinutesSecondsFrames(timeAttributeValue, rateInfo);
    } else if (TIME_UNIT_REGEX.test(timeAttributeValue)) {
      seconds = parseTimeUnits(timeAttributeValue, rateInfo);
    }
  }
  return seconds;
}
function parseHoursMinutesSecondsFrames(timeAttributeValue, rateInfo) {
  const m = HMSF_REGEX.exec(timeAttributeValue);
  const frames = (m[4] | 0) + (m[5] | 0) / rateInfo.subFrameRate;
  return (m[1] | 0) * 3600 + (m[2] | 0) * 60 + (m[3] | 0) + frames / rateInfo.frameRate;
}
function parseTimeUnits(timeAttributeValue, rateInfo) {
  const m = TIME_UNIT_REGEX.exec(timeAttributeValue);
  const value = Number(m[1]);
  const unit = m[2];
  switch (unit) {
    case "h":
      return value * 3600;
    case "m":
      return value * 60;
    case "ms":
      return value * 1e3;
    case "f":
      return value / rateInfo.frameRate;
    case "t":
      return value / rateInfo.tickRate;
  }
  return value;
}
class TimelineController {
  constructor(hls) {
    this.hls = void 0;
    this.media = null;
    this.config = void 0;
    this.enabled = true;
    this.Cues = void 0;
    this.textTracks = [];
    this.tracks = [];
    this.initPTS = [];
    this.unparsedVttFrags = [];
    this.captionsTracks = {};
    this.nonNativeCaptionsTracks = {};
    this.cea608Parser1 = void 0;
    this.cea608Parser2 = void 0;
    this.lastCc = -1;
    this.lastSn = -1;
    this.lastPartIndex = -1;
    this.prevCC = -1;
    this.vttCCs = newVTTCCs();
    this.captionsProperties = void 0;
    this.hls = hls;
    this.config = hls.config;
    this.Cues = hls.config.cueHandler;
    this.captionsProperties = {
      textTrack1: {
        label: this.config.captionsTextTrack1Label,
        languageCode: this.config.captionsTextTrack1LanguageCode
      },
      textTrack2: {
        label: this.config.captionsTextTrack2Label,
        languageCode: this.config.captionsTextTrack2LanguageCode
      },
      textTrack3: {
        label: this.config.captionsTextTrack3Label,
        languageCode: this.config.captionsTextTrack3LanguageCode
      },
      textTrack4: {
        label: this.config.captionsTextTrack4Label,
        languageCode: this.config.captionsTextTrack4LanguageCode
      }
    };
    hls.on(Events.MEDIA_ATTACHING, this.onMediaAttaching, this);
    hls.on(Events.MEDIA_DETACHING, this.onMediaDetaching, this);
    hls.on(Events.MANIFEST_LOADING, this.onManifestLoading, this);
    hls.on(Events.MANIFEST_LOADED, this.onManifestLoaded, this);
    hls.on(Events.SUBTITLE_TRACKS_UPDATED, this.onSubtitleTracksUpdated, this);
    hls.on(Events.FRAG_LOADING, this.onFragLoading, this);
    hls.on(Events.FRAG_LOADED, this.onFragLoaded, this);
    hls.on(Events.FRAG_PARSING_USERDATA, this.onFragParsingUserdata, this);
    hls.on(Events.FRAG_DECRYPTED, this.onFragDecrypted, this);
    hls.on(Events.INIT_PTS_FOUND, this.onInitPtsFound, this);
    hls.on(Events.SUBTITLE_TRACKS_CLEARED, this.onSubtitleTracksCleared, this);
    hls.on(Events.BUFFER_FLUSHING, this.onBufferFlushing, this);
  }
  destroy() {
    const {
      hls
    } = this;
    hls.off(Events.MEDIA_ATTACHING, this.onMediaAttaching, this);
    hls.off(Events.MEDIA_DETACHING, this.onMediaDetaching, this);
    hls.off(Events.MANIFEST_LOADING, this.onManifestLoading, this);
    hls.off(Events.MANIFEST_LOADED, this.onManifestLoaded, this);
    hls.off(Events.SUBTITLE_TRACKS_UPDATED, this.onSubtitleTracksUpdated, this);
    hls.off(Events.FRAG_LOADING, this.onFragLoading, this);
    hls.off(Events.FRAG_LOADED, this.onFragLoaded, this);
    hls.off(Events.FRAG_PARSING_USERDATA, this.onFragParsingUserdata, this);
    hls.off(Events.FRAG_DECRYPTED, this.onFragDecrypted, this);
    hls.off(Events.INIT_PTS_FOUND, this.onInitPtsFound, this);
    hls.off(Events.SUBTITLE_TRACKS_CLEARED, this.onSubtitleTracksCleared, this);
    hls.off(Events.BUFFER_FLUSHING, this.onBufferFlushing, this);
    this.hls = this.config = null;
    this.cea608Parser1 = this.cea608Parser2 = void 0;
  }
  initCea608Parsers() {
    if (this.config.enableCEA708Captions && (!this.cea608Parser1 || !this.cea608Parser2)) {
      const channel1 = new OutputFilter(this, "textTrack1");
      const channel2 = new OutputFilter(this, "textTrack2");
      const channel3 = new OutputFilter(this, "textTrack3");
      const channel4 = new OutputFilter(this, "textTrack4");
      this.cea608Parser1 = new Cea608Parser(1, channel1, channel2);
      this.cea608Parser2 = new Cea608Parser(3, channel3, channel4);
    }
  }
  addCues(trackName, startTime, endTime, screen, cueRanges) {
    let merged = false;
    for (let i2 = cueRanges.length; i2--; ) {
      const cueRange = cueRanges[i2];
      const overlap = intersection(cueRange[0], cueRange[1], startTime, endTime);
      if (overlap >= 0) {
        cueRange[0] = Math.min(cueRange[0], startTime);
        cueRange[1] = Math.max(cueRange[1], endTime);
        merged = true;
        if (overlap / (endTime - startTime) > 0.5) {
          return;
        }
      }
    }
    if (!merged) {
      cueRanges.push([startTime, endTime]);
    }
    if (this.config.renderTextTracksNatively) {
      const track = this.captionsTracks[trackName];
      this.Cues.newCue(track, startTime, endTime, screen);
    } else {
      const cues = this.Cues.newCue(null, startTime, endTime, screen);
      this.hls.trigger(Events.CUES_PARSED, {
        type: "captions",
        cues,
        track: trackName
      });
    }
  }
  // Triggered when an initial PTS is found; used for synchronisation of WebVTT.
  onInitPtsFound(event, {
    frag,
    id,
    initPTS,
    timescale
  }) {
    const {
      unparsedVttFrags
    } = this;
    if (id === "main") {
      this.initPTS[frag.cc] = {
        baseTime: initPTS,
        timescale
      };
    }
    if (unparsedVttFrags.length) {
      this.unparsedVttFrags = [];
      unparsedVttFrags.forEach((frag2) => {
        this.onFragLoaded(Events.FRAG_LOADED, frag2);
      });
    }
  }
  getExistingTrack(label, language) {
    const {
      media
    } = this;
    if (media) {
      for (let i2 = 0; i2 < media.textTracks.length; i2++) {
        const textTrack = media.textTracks[i2];
        if (canReuseVttTextTrack(textTrack, {
          name: label,
          lang: language
        })) {
          return textTrack;
        }
      }
    }
    return null;
  }
  createCaptionsTrack(trackName) {
    if (this.config.renderTextTracksNatively) {
      this.createNativeTrack(trackName);
    } else {
      this.createNonNativeTrack(trackName);
    }
  }
  createNativeTrack(trackName) {
    if (this.captionsTracks[trackName]) {
      return;
    }
    const {
      captionsProperties,
      captionsTracks,
      media
    } = this;
    const {
      label,
      languageCode
    } = captionsProperties[trackName];
    const existingTrack = this.getExistingTrack(label, languageCode);
    if (!existingTrack) {
      const textTrack = this.createTextTrack("captions", label, languageCode);
      if (textTrack) {
        textTrack[trackName] = true;
        captionsTracks[trackName] = textTrack;
      }
    } else {
      captionsTracks[trackName] = existingTrack;
      clearCurrentCues(captionsTracks[trackName]);
      sendAddTrackEvent(captionsTracks[trackName], media);
    }
  }
  createNonNativeTrack(trackName) {
    if (this.nonNativeCaptionsTracks[trackName]) {
      return;
    }
    const trackProperties = this.captionsProperties[trackName];
    if (!trackProperties) {
      return;
    }
    const label = trackProperties.label;
    const track = {
      _id: trackName,
      label,
      kind: "captions",
      default: trackProperties.media ? !!trackProperties.media.default : false,
      closedCaptions: trackProperties.media
    };
    this.nonNativeCaptionsTracks[trackName] = track;
    this.hls.trigger(Events.NON_NATIVE_TEXT_TRACKS_FOUND, {
      tracks: [track]
    });
  }
  createTextTrack(kind, label, lang) {
    const media = this.media;
    if (!media) {
      return;
    }
    return media.addTextTrack(kind, label, lang);
  }
  onMediaAttaching(event, data) {
    this.media = data.media;
    this._cleanTracks();
  }
  onMediaDetaching() {
    const {
      captionsTracks
    } = this;
    Object.keys(captionsTracks).forEach((trackName) => {
      clearCurrentCues(captionsTracks[trackName]);
      delete captionsTracks[trackName];
    });
    this.nonNativeCaptionsTracks = {};
  }
  onManifestLoading() {
    this.lastCc = -1;
    this.lastSn = -1;
    this.lastPartIndex = -1;
    this.prevCC = -1;
    this.vttCCs = newVTTCCs();
    this._cleanTracks();
    this.tracks = [];
    this.captionsTracks = {};
    this.nonNativeCaptionsTracks = {};
    this.textTracks = [];
    this.unparsedVttFrags = [];
    this.initPTS = [];
    if (this.cea608Parser1 && this.cea608Parser2) {
      this.cea608Parser1.reset();
      this.cea608Parser2.reset();
    }
  }
  _cleanTracks() {
    const {
      media
    } = this;
    if (!media) {
      return;
    }
    const textTracks = media.textTracks;
    if (textTracks) {
      for (let i2 = 0; i2 < textTracks.length; i2++) {
        clearCurrentCues(textTracks[i2]);
      }
    }
  }
  onSubtitleTracksUpdated(event, data) {
    const tracks = data.subtitleTracks || [];
    const hasIMSC1 = tracks.some((track) => track.textCodec === IMSC1_CODEC);
    if (this.config.enableWebVTT || hasIMSC1 && this.config.enableIMSC1) {
      const listIsIdentical = subtitleOptionsIdentical(this.tracks, tracks);
      if (listIsIdentical) {
        this.tracks = tracks;
        return;
      }
      this.textTracks = [];
      this.tracks = tracks;
      if (this.config.renderTextTracksNatively) {
        const media = this.media;
        const inUseTracks = media ? filterSubtitleTracks(media.textTracks) : null;
        this.tracks.forEach((track, index) => {
          let textTrack;
          if (inUseTracks) {
            let inUseTrack = null;
            for (let i2 = 0; i2 < inUseTracks.length; i2++) {
              if (inUseTracks[i2] && canReuseVttTextTrack(inUseTracks[i2], track)) {
                inUseTrack = inUseTracks[i2];
                inUseTracks[i2] = null;
                break;
              }
            }
            if (inUseTrack) {
              textTrack = inUseTrack;
            }
          }
          if (textTrack) {
            clearCurrentCues(textTrack);
          } else {
            const textTrackKind = captionsOrSubtitlesFromCharacteristics(track);
            textTrack = this.createTextTrack(textTrackKind, track.name, track.lang);
            if (textTrack) {
              textTrack.mode = "disabled";
            }
          }
          if (textTrack) {
            this.textTracks.push(textTrack);
          }
        });
        if (inUseTracks != null && inUseTracks.length) {
          const unusedTextTracks = inUseTracks.filter((t2) => t2 !== null).map((t2) => t2.label);
          if (unusedTextTracks.length) {
            logger.warn(`Media element contains unused subtitle tracks: ${unusedTextTracks.join(", ")}. Replace media element for each source to clear TextTracks and captions menu.`);
          }
        }
      } else if (this.tracks.length) {
        const tracksList = this.tracks.map((track) => {
          return {
            label: track.name,
            kind: track.type.toLowerCase(),
            default: track.default,
            subtitleTrack: track
          };
        });
        this.hls.trigger(Events.NON_NATIVE_TEXT_TRACKS_FOUND, {
          tracks: tracksList
        });
      }
    }
  }
  onManifestLoaded(event, data) {
    if (this.config.enableCEA708Captions && data.captions) {
      data.captions.forEach((captionsTrack) => {
        const instreamIdMatch = /(?:CC|SERVICE)([1-4])/.exec(captionsTrack.instreamId);
        if (!instreamIdMatch) {
          return;
        }
        const trackName = `textTrack${instreamIdMatch[1]}`;
        const trackProperties = this.captionsProperties[trackName];
        if (!trackProperties) {
          return;
        }
        trackProperties.label = captionsTrack.name;
        if (captionsTrack.lang) {
          trackProperties.languageCode = captionsTrack.lang;
        }
        trackProperties.media = captionsTrack;
      });
    }
  }
  closedCaptionsForLevel(frag) {
    const level = this.hls.levels[frag.level];
    return level == null ? void 0 : level.attrs["CLOSED-CAPTIONS"];
  }
  onFragLoading(event, data) {
    if (this.enabled && data.frag.type === PlaylistLevelType.MAIN) {
      var _data$part$index, _data$part;
      const {
        cea608Parser1,
        cea608Parser2,
        lastSn
      } = this;
      const {
        cc,
        sn
      } = data.frag;
      const partIndex = (_data$part$index = (_data$part = data.part) == null ? void 0 : _data$part.index) != null ? _data$part$index : -1;
      if (cea608Parser1 && cea608Parser2) {
        if (sn !== lastSn + 1 || sn === lastSn && partIndex !== this.lastPartIndex + 1 || cc !== this.lastCc) {
          cea608Parser1.reset();
          cea608Parser2.reset();
        }
      }
      this.lastCc = cc;
      this.lastSn = sn;
      this.lastPartIndex = partIndex;
    }
  }
  onFragLoaded(event, data) {
    const {
      frag,
      payload
    } = data;
    if (frag.type === PlaylistLevelType.SUBTITLE) {
      if (payload.byteLength) {
        const decryptData = frag.decryptdata;
        const decrypted = "stats" in data;
        if (decryptData == null || !decryptData.encrypted || decrypted) {
          const trackPlaylistMedia = this.tracks[frag.level];
          const vttCCs = this.vttCCs;
          if (!vttCCs[frag.cc]) {
            vttCCs[frag.cc] = {
              start: frag.start,
              prevCC: this.prevCC,
              new: true
            };
            this.prevCC = frag.cc;
          }
          if (trackPlaylistMedia && trackPlaylistMedia.textCodec === IMSC1_CODEC) {
            this._parseIMSC1(frag, payload);
          } else {
            this._parseVTTs(data);
          }
        }
      } else {
        this.hls.trigger(Events.SUBTITLE_FRAG_PROCESSED, {
          success: false,
          frag,
          error: new Error("Empty subtitle payload")
        });
      }
    }
  }
  _parseIMSC1(frag, payload) {
    const hls = this.hls;
    parseIMSC1(payload, this.initPTS[frag.cc], (cues) => {
      this._appendCues(cues, frag.level);
      hls.trigger(Events.SUBTITLE_FRAG_PROCESSED, {
        success: true,
        frag
      });
    }, (error) => {
      logger.log(`Failed to parse IMSC1: ${error}`);
      hls.trigger(Events.SUBTITLE_FRAG_PROCESSED, {
        success: false,
        frag,
        error
      });
    });
  }
  _parseVTTs(data) {
    var _frag$initSegment;
    const {
      frag,
      payload
    } = data;
    const {
      initPTS,
      unparsedVttFrags
    } = this;
    const maxAvCC = initPTS.length - 1;
    if (!initPTS[frag.cc] && maxAvCC === -1) {
      unparsedVttFrags.push(data);
      return;
    }
    const hls = this.hls;
    const payloadWebVTT = (_frag$initSegment = frag.initSegment) != null && _frag$initSegment.data ? appendUint8Array(frag.initSegment.data, new Uint8Array(payload)) : payload;
    parseWebVTT(payloadWebVTT, this.initPTS[frag.cc], this.vttCCs, frag.cc, frag.start, (cues) => {
      this._appendCues(cues, frag.level);
      hls.trigger(Events.SUBTITLE_FRAG_PROCESSED, {
        success: true,
        frag
      });
    }, (error) => {
      const missingInitPTS = error.message === "Missing initPTS for VTT MPEGTS";
      if (missingInitPTS) {
        unparsedVttFrags.push(data);
      } else {
        this._fallbackToIMSC1(frag, payload);
      }
      logger.log(`Failed to parse VTT cue: ${error}`);
      if (missingInitPTS && maxAvCC > frag.cc) {
        return;
      }
      hls.trigger(Events.SUBTITLE_FRAG_PROCESSED, {
        success: false,
        frag,
        error
      });
    });
  }
  _fallbackToIMSC1(frag, payload) {
    const trackPlaylistMedia = this.tracks[frag.level];
    if (!trackPlaylistMedia.textCodec) {
      parseIMSC1(payload, this.initPTS[frag.cc], () => {
        trackPlaylistMedia.textCodec = IMSC1_CODEC;
        this._parseIMSC1(frag, payload);
      }, () => {
        trackPlaylistMedia.textCodec = "wvtt";
      });
    }
  }
  _appendCues(cues, fragLevel) {
    const hls = this.hls;
    if (this.config.renderTextTracksNatively) {
      const textTrack = this.textTracks[fragLevel];
      if (!textTrack || textTrack.mode === "disabled") {
        return;
      }
      cues.forEach((cue) => addCueToTrack(textTrack, cue));
    } else {
      const currentTrack = this.tracks[fragLevel];
      if (!currentTrack) {
        return;
      }
      const track = currentTrack.default ? "default" : "subtitles" + fragLevel;
      hls.trigger(Events.CUES_PARSED, {
        type: "subtitles",
        cues,
        track
      });
    }
  }
  onFragDecrypted(event, data) {
    const {
      frag
    } = data;
    if (frag.type === PlaylistLevelType.SUBTITLE) {
      this.onFragLoaded(Events.FRAG_LOADED, data);
    }
  }
  onSubtitleTracksCleared() {
    this.tracks = [];
    this.captionsTracks = {};
  }
  onFragParsingUserdata(event, data) {
    this.initCea608Parsers();
    const {
      cea608Parser1,
      cea608Parser2
    } = this;
    if (!this.enabled || !cea608Parser1 || !cea608Parser2) {
      return;
    }
    const {
      frag,
      samples
    } = data;
    if (frag.type === PlaylistLevelType.MAIN && this.closedCaptionsForLevel(frag) === "NONE") {
      return;
    }
    for (let i2 = 0; i2 < samples.length; i2++) {
      const ccBytes = samples[i2].bytes;
      if (ccBytes) {
        const ccdatas = this.extractCea608Data(ccBytes);
        cea608Parser1.addData(samples[i2].pts, ccdatas[0]);
        cea608Parser2.addData(samples[i2].pts, ccdatas[1]);
      }
    }
  }
  onBufferFlushing(event, {
    startOffset,
    endOffset,
    endOffsetSubtitles,
    type
  }) {
    const {
      media
    } = this;
    if (!media || media.currentTime < endOffset) {
      return;
    }
    if (!type || type === "video") {
      const {
        captionsTracks
      } = this;
      Object.keys(captionsTracks).forEach((trackName) => removeCuesInRange(captionsTracks[trackName], startOffset, endOffset));
    }
    if (this.config.renderTextTracksNatively) {
      if (startOffset === 0 && endOffsetSubtitles !== void 0) {
        const {
          textTracks
        } = this;
        Object.keys(textTracks).forEach((trackName) => removeCuesInRange(textTracks[trackName], startOffset, endOffsetSubtitles));
      }
    }
  }
  extractCea608Data(byteArray) {
    const actualCCBytes = [[], []];
    const count = byteArray[0] & 31;
    let position = 2;
    for (let j = 0; j < count; j++) {
      const tmpByte = byteArray[position++];
      const ccbyte1 = 127 & byteArray[position++];
      const ccbyte2 = 127 & byteArray[position++];
      if (ccbyte1 === 0 && ccbyte2 === 0) {
        continue;
      }
      const ccValid = (4 & tmpByte) !== 0;
      if (ccValid) {
        const ccType = 3 & tmpByte;
        if (0 === ccType || 1 === ccType) {
          actualCCBytes[ccType].push(ccbyte1);
          actualCCBytes[ccType].push(ccbyte2);
        }
      }
    }
    return actualCCBytes;
  }
}
function captionsOrSubtitlesFromCharacteristics(track) {
  if (track.characteristics) {
    if (/transcribes-spoken-dialog/gi.test(track.characteristics) && /describes-music-and-sound/gi.test(track.characteristics)) {
      return "captions";
    }
  }
  return "subtitles";
}
function canReuseVttTextTrack(inUseTrack, manifestTrack) {
  return !!inUseTrack && inUseTrack.kind === captionsOrSubtitlesFromCharacteristics(manifestTrack) && subtitleTrackMatchesTextTrack(manifestTrack, inUseTrack);
}
function intersection(x1, x2, y1, y2) {
  return Math.min(x2, y2) - Math.max(x1, y1);
}
function newVTTCCs() {
  return {
    ccOffset: 0,
    presentationOffset: 0,
    0: {
      start: 0,
      prevCC: -1,
      new: true
    }
  };
}
class CapLevelController {
  constructor(hls) {
    this.hls = void 0;
    this.autoLevelCapping = void 0;
    this.firstLevel = void 0;
    this.media = void 0;
    this.restrictedLevels = void 0;
    this.timer = void 0;
    this.clientRect = void 0;
    this.streamController = void 0;
    this.hls = hls;
    this.autoLevelCapping = Number.POSITIVE_INFINITY;
    this.firstLevel = -1;
    this.media = null;
    this.restrictedLevels = [];
    this.timer = void 0;
    this.clientRect = null;
    this.registerListeners();
  }
  setStreamController(streamController) {
    this.streamController = streamController;
  }
  destroy() {
    if (this.hls) {
      this.unregisterListener();
    }
    if (this.timer) {
      this.stopCapping();
    }
    this.media = null;
    this.clientRect = null;
    this.hls = this.streamController = null;
  }
  registerListeners() {
    const {
      hls
    } = this;
    hls.on(Events.FPS_DROP_LEVEL_CAPPING, this.onFpsDropLevelCapping, this);
    hls.on(Events.MEDIA_ATTACHING, this.onMediaAttaching, this);
    hls.on(Events.MANIFEST_PARSED, this.onManifestParsed, this);
    hls.on(Events.LEVELS_UPDATED, this.onLevelsUpdated, this);
    hls.on(Events.BUFFER_CODECS, this.onBufferCodecs, this);
    hls.on(Events.MEDIA_DETACHING, this.onMediaDetaching, this);
  }
  unregisterListener() {
    const {
      hls
    } = this;
    hls.off(Events.FPS_DROP_LEVEL_CAPPING, this.onFpsDropLevelCapping, this);
    hls.off(Events.MEDIA_ATTACHING, this.onMediaAttaching, this);
    hls.off(Events.MANIFEST_PARSED, this.onManifestParsed, this);
    hls.off(Events.LEVELS_UPDATED, this.onLevelsUpdated, this);
    hls.off(Events.BUFFER_CODECS, this.onBufferCodecs, this);
    hls.off(Events.MEDIA_DETACHING, this.onMediaDetaching, this);
  }
  onFpsDropLevelCapping(event, data) {
    const level = this.hls.levels[data.droppedLevel];
    if (this.isLevelAllowed(level)) {
      this.restrictedLevels.push({
        bitrate: level.bitrate,
        height: level.height,
        width: level.width
      });
    }
  }
  onMediaAttaching(event, data) {
    this.media = data.media instanceof HTMLVideoElement ? data.media : null;
    this.clientRect = null;
    if (this.timer && this.hls.levels.length) {
      this.detectPlayerSize();
    }
  }
  onManifestParsed(event, data) {
    const hls = this.hls;
    this.restrictedLevels = [];
    this.firstLevel = data.firstLevel;
    if (hls.config.capLevelToPlayerSize && data.video) {
      this.startCapping();
    }
  }
  onLevelsUpdated(event, data) {
    if (this.timer && isFiniteNumber(this.autoLevelCapping)) {
      this.detectPlayerSize();
    }
  }
  // Only activate capping when playing a video stream; otherwise, multi-bitrate audio-only streams will be restricted
  // to the first level
  onBufferCodecs(event, data) {
    const hls = this.hls;
    if (hls.config.capLevelToPlayerSize && data.video) {
      this.startCapping();
    }
  }
  onMediaDetaching() {
    this.stopCapping();
  }
  detectPlayerSize() {
    if (this.media) {
      if (this.mediaHeight <= 0 || this.mediaWidth <= 0) {
        this.clientRect = null;
        return;
      }
      const levels = this.hls.levels;
      if (levels.length) {
        const hls = this.hls;
        const maxLevel = this.getMaxLevel(levels.length - 1);
        if (maxLevel !== this.autoLevelCapping) {
          logger.log(`Setting autoLevelCapping to ${maxLevel}: ${levels[maxLevel].height}p@${levels[maxLevel].bitrate} for media ${this.mediaWidth}x${this.mediaHeight}`);
        }
        hls.autoLevelCapping = maxLevel;
        if (hls.autoLevelCapping > this.autoLevelCapping && this.streamController) {
          this.streamController.nextLevelSwitch();
        }
        this.autoLevelCapping = hls.autoLevelCapping;
      }
    }
  }
  /*
   * returns level should be the one with the dimensions equal or greater than the media (player) dimensions (so the video will be downscaled)
   */
  getMaxLevel(capLevelIndex) {
    const levels = this.hls.levels;
    if (!levels.length) {
      return -1;
    }
    const validLevels = levels.filter((level, index) => this.isLevelAllowed(level) && index <= capLevelIndex);
    this.clientRect = null;
    return CapLevelController.getMaxLevelByMediaSize(validLevels, this.mediaWidth, this.mediaHeight);
  }
  startCapping() {
    if (this.timer) {
      return;
    }
    this.autoLevelCapping = Number.POSITIVE_INFINITY;
    self.clearInterval(this.timer);
    this.timer = self.setInterval(this.detectPlayerSize.bind(this), 1e3);
    this.detectPlayerSize();
  }
  stopCapping() {
    this.restrictedLevels = [];
    this.firstLevel = -1;
    this.autoLevelCapping = Number.POSITIVE_INFINITY;
    if (this.timer) {
      self.clearInterval(this.timer);
      this.timer = void 0;
    }
  }
  getDimensions() {
    if (this.clientRect) {
      return this.clientRect;
    }
    const media = this.media;
    const boundsRect = {
      width: 0,
      height: 0
    };
    if (media) {
      const clientRect = media.getBoundingClientRect();
      boundsRect.width = clientRect.width;
      boundsRect.height = clientRect.height;
      if (!boundsRect.width && !boundsRect.height) {
        boundsRect.width = clientRect.right - clientRect.left || media.width || 0;
        boundsRect.height = clientRect.bottom - clientRect.top || media.height || 0;
      }
    }
    this.clientRect = boundsRect;
    return boundsRect;
  }
  get mediaWidth() {
    return this.getDimensions().width * this.contentScaleFactor;
  }
  get mediaHeight() {
    return this.getDimensions().height * this.contentScaleFactor;
  }
  get contentScaleFactor() {
    let pixelRatio = 1;
    if (!this.hls.config.ignoreDevicePixelRatio) {
      try {
        pixelRatio = self.devicePixelRatio;
      } catch (e2) {
      }
    }
    return pixelRatio;
  }
  isLevelAllowed(level) {
    const restrictedLevels = this.restrictedLevels;
    return !restrictedLevels.some((restrictedLevel) => {
      return level.bitrate === restrictedLevel.bitrate && level.width === restrictedLevel.width && level.height === restrictedLevel.height;
    });
  }
  static getMaxLevelByMediaSize(levels, width, height) {
    if (!(levels != null && levels.length)) {
      return -1;
    }
    const atGreatestBandwidth = (curLevel, nextLevel) => {
      if (!nextLevel) {
        return true;
      }
      return curLevel.width !== nextLevel.width || curLevel.height !== nextLevel.height;
    };
    let maxLevelIndex = levels.length - 1;
    const squareSize = Math.max(width, height);
    for (let i2 = 0; i2 < levels.length; i2 += 1) {
      const level = levels[i2];
      if ((level.width >= squareSize || level.height >= squareSize) && atGreatestBandwidth(level, levels[i2 + 1])) {
        maxLevelIndex = i2;
        break;
      }
    }
    return maxLevelIndex;
  }
}
class FPSController {
  constructor(hls) {
    this.hls = void 0;
    this.isVideoPlaybackQualityAvailable = false;
    this.timer = void 0;
    this.media = null;
    this.lastTime = void 0;
    this.lastDroppedFrames = 0;
    this.lastDecodedFrames = 0;
    this.streamController = void 0;
    this.hls = hls;
    this.registerListeners();
  }
  setStreamController(streamController) {
    this.streamController = streamController;
  }
  registerListeners() {
    this.hls.on(Events.MEDIA_ATTACHING, this.onMediaAttaching, this);
  }
  unregisterListeners() {
    this.hls.off(Events.MEDIA_ATTACHING, this.onMediaAttaching, this);
  }
  destroy() {
    if (this.timer) {
      clearInterval(this.timer);
    }
    this.unregisterListeners();
    this.isVideoPlaybackQualityAvailable = false;
    this.media = null;
  }
  onMediaAttaching(event, data) {
    const config = this.hls.config;
    if (config.capLevelOnFPSDrop) {
      const media = data.media instanceof self.HTMLVideoElement ? data.media : null;
      this.media = media;
      if (media && typeof media.getVideoPlaybackQuality === "function") {
        this.isVideoPlaybackQualityAvailable = true;
      }
      self.clearInterval(this.timer);
      this.timer = self.setInterval(this.checkFPSInterval.bind(this), config.fpsDroppedMonitoringPeriod);
    }
  }
  checkFPS(video, decodedFrames, droppedFrames) {
    const currentTime = performance.now();
    if (decodedFrames) {
      if (this.lastTime) {
        const currentPeriod = currentTime - this.lastTime;
        const currentDropped = droppedFrames - this.lastDroppedFrames;
        const currentDecoded = decodedFrames - this.lastDecodedFrames;
        const droppedFPS = 1e3 * currentDropped / currentPeriod;
        const hls = this.hls;
        hls.trigger(Events.FPS_DROP, {
          currentDropped,
          currentDecoded,
          totalDroppedFrames: droppedFrames
        });
        if (droppedFPS > 0) {
          if (currentDropped > hls.config.fpsDroppedMonitoringThreshold * currentDecoded) {
            let currentLevel = hls.currentLevel;
            logger.warn("drop FPS ratio greater than max allowed value for currentLevel: " + currentLevel);
            if (currentLevel > 0 && (hls.autoLevelCapping === -1 || hls.autoLevelCapping >= currentLevel)) {
              currentLevel = currentLevel - 1;
              hls.trigger(Events.FPS_DROP_LEVEL_CAPPING, {
                level: currentLevel,
                droppedLevel: hls.currentLevel
              });
              hls.autoLevelCapping = currentLevel;
              this.streamController.nextLevelSwitch();
            }
          }
        }
      }
      this.lastTime = currentTime;
      this.lastDroppedFrames = droppedFrames;
      this.lastDecodedFrames = decodedFrames;
    }
  }
  checkFPSInterval() {
    const video = this.media;
    if (video) {
      if (this.isVideoPlaybackQualityAvailable) {
        const videoPlaybackQuality = video.getVideoPlaybackQuality();
        this.checkFPS(video, videoPlaybackQuality.totalVideoFrames, videoPlaybackQuality.droppedVideoFrames);
      } else {
        this.checkFPS(video, video.webkitDecodedFrameCount, video.webkitDroppedFrameCount);
      }
    }
  }
}
const LOGGER_PREFIX = "[eme]";
class EMEController {
  constructor(hls) {
    this.hls = void 0;
    this.config = void 0;
    this.media = null;
    this.keyFormatPromise = null;
    this.keySystemAccessPromises = {};
    this._requestLicenseFailureCount = 0;
    this.mediaKeySessions = [];
    this.keyIdToKeySessionPromise = {};
    this.setMediaKeysQueue = EMEController.CDMCleanupPromise ? [EMEController.CDMCleanupPromise] : [];
    this.onMediaEncrypted = this._onMediaEncrypted.bind(this);
    this.onWaitingForKey = this._onWaitingForKey.bind(this);
    this.debug = logger.debug.bind(logger, LOGGER_PREFIX);
    this.log = logger.log.bind(logger, LOGGER_PREFIX);
    this.warn = logger.warn.bind(logger, LOGGER_PREFIX);
    this.error = logger.error.bind(logger, LOGGER_PREFIX);
    this.hls = hls;
    this.config = hls.config;
    this.registerListeners();
  }
  destroy() {
    this.unregisterListeners();
    this.onMediaDetached();
    const config = this.config;
    config.requestMediaKeySystemAccessFunc = null;
    config.licenseXhrSetup = config.licenseResponseCallback = void 0;
    config.drmSystems = config.drmSystemOptions = {};
    this.hls = this.onMediaEncrypted = this.onWaitingForKey = this.keyIdToKeySessionPromise = null;
    this.config = null;
  }
  registerListeners() {
    this.hls.on(Events.MEDIA_ATTACHED, this.onMediaAttached, this);
    this.hls.on(Events.MEDIA_DETACHED, this.onMediaDetached, this);
    this.hls.on(Events.MANIFEST_LOADING, this.onManifestLoading, this);
    this.hls.on(Events.MANIFEST_LOADED, this.onManifestLoaded, this);
  }
  unregisterListeners() {
    this.hls.off(Events.MEDIA_ATTACHED, this.onMediaAttached, this);
    this.hls.off(Events.MEDIA_DETACHED, this.onMediaDetached, this);
    this.hls.off(Events.MANIFEST_LOADING, this.onManifestLoading, this);
    this.hls.off(Events.MANIFEST_LOADED, this.onManifestLoaded, this);
  }
  getLicenseServerUrl(keySystem) {
    const {
      drmSystems,
      widevineLicenseUrl
    } = this.config;
    const keySystemConfiguration = drmSystems[keySystem];
    if (keySystemConfiguration) {
      return keySystemConfiguration.licenseUrl;
    }
    if (keySystem === KeySystems.WIDEVINE && widevineLicenseUrl) {
      return widevineLicenseUrl;
    }
    throw new Error(`no license server URL configured for key-system "${keySystem}"`);
  }
  getServerCertificateUrl(keySystem) {
    const {
      drmSystems
    } = this.config;
    const keySystemConfiguration = drmSystems[keySystem];
    if (keySystemConfiguration) {
      return keySystemConfiguration.serverCertificateUrl;
    } else {
      this.log(`No Server Certificate in config.drmSystems["${keySystem}"]`);
    }
  }
  attemptKeySystemAccess(keySystemsToAttempt) {
    const levels = this.hls.levels;
    const uniqueCodec = (value, i2, a) => !!value && a.indexOf(value) === i2;
    const audioCodecs = levels.map((level) => level.audioCodec).filter(uniqueCodec);
    const videoCodecs = levels.map((level) => level.videoCodec).filter(uniqueCodec);
    if (audioCodecs.length + videoCodecs.length === 0) {
      videoCodecs.push("avc1.42e01e");
    }
    return new Promise((resolve, reject) => {
      const attempt = (keySystems) => {
        const keySystem = keySystems.shift();
        this.getMediaKeysPromise(keySystem, audioCodecs, videoCodecs).then((mediaKeys) => resolve({
          keySystem,
          mediaKeys
        })).catch((error) => {
          if (keySystems.length) {
            attempt(keySystems);
          } else if (error instanceof EMEKeyError) {
            reject(error);
          } else {
            reject(new EMEKeyError({
              type: ErrorTypes.KEY_SYSTEM_ERROR,
              details: ErrorDetails.KEY_SYSTEM_NO_ACCESS,
              error,
              fatal: true
            }, error.message));
          }
        });
      };
      attempt(keySystemsToAttempt);
    });
  }
  requestMediaKeySystemAccess(keySystem, supportedConfigurations) {
    const {
      requestMediaKeySystemAccessFunc
    } = this.config;
    if (!(typeof requestMediaKeySystemAccessFunc === "function")) {
      let errMessage = `Configured requestMediaKeySystemAccess is not a function ${requestMediaKeySystemAccessFunc}`;
      if (requestMediaKeySystemAccess === null && self.location.protocol === "http:") {
        errMessage = `navigator.requestMediaKeySystemAccess is not available over insecure protocol ${location.protocol}`;
      }
      return Promise.reject(new Error(errMessage));
    }
    return requestMediaKeySystemAccessFunc(keySystem, supportedConfigurations);
  }
  getMediaKeysPromise(keySystem, audioCodecs, videoCodecs) {
    const mediaKeySystemConfigs = getSupportedMediaKeySystemConfigurations(keySystem, audioCodecs, videoCodecs, this.config.drmSystemOptions);
    const keySystemAccessPromises = this.keySystemAccessPromises[keySystem];
    let keySystemAccess = keySystemAccessPromises == null ? void 0 : keySystemAccessPromises.keySystemAccess;
    if (!keySystemAccess) {
      this.log(`Requesting encrypted media "${keySystem}" key-system access with config: ${JSON.stringify(mediaKeySystemConfigs)}`);
      keySystemAccess = this.requestMediaKeySystemAccess(keySystem, mediaKeySystemConfigs);
      const _keySystemAccessPromises = this.keySystemAccessPromises[keySystem] = {
        keySystemAccess
      };
      keySystemAccess.catch((error) => {
        this.log(`Failed to obtain access to key-system "${keySystem}": ${error}`);
      });
      return keySystemAccess.then((mediaKeySystemAccess) => {
        this.log(`Access for key-system "${mediaKeySystemAccess.keySystem}" obtained`);
        const certificateRequest = this.fetchServerCertificate(keySystem);
        this.log(`Create media-keys for "${keySystem}"`);
        _keySystemAccessPromises.mediaKeys = mediaKeySystemAccess.createMediaKeys().then((mediaKeys) => {
          this.log(`Media-keys created for "${keySystem}"`);
          return certificateRequest.then((certificate) => {
            if (certificate) {
              return this.setMediaKeysServerCertificate(mediaKeys, keySystem, certificate);
            }
            return mediaKeys;
          });
        });
        _keySystemAccessPromises.mediaKeys.catch((error) => {
          this.error(`Failed to create media-keys for "${keySystem}"}: ${error}`);
        });
        return _keySystemAccessPromises.mediaKeys;
      });
    }
    return keySystemAccess.then(() => keySystemAccessPromises.mediaKeys);
  }
  createMediaKeySessionContext({
    decryptdata,
    keySystem,
    mediaKeys
  }) {
    this.log(`Creating key-system session "${keySystem}" keyId: ${Hex.hexDump(decryptdata.keyId || [])}`);
    const mediaKeysSession = mediaKeys.createSession();
    const mediaKeySessionContext = {
      decryptdata,
      keySystem,
      mediaKeys,
      mediaKeysSession,
      keyStatus: "status-pending"
    };
    this.mediaKeySessions.push(mediaKeySessionContext);
    return mediaKeySessionContext;
  }
  renewKeySession(mediaKeySessionContext) {
    const decryptdata = mediaKeySessionContext.decryptdata;
    if (decryptdata.pssh) {
      const keySessionContext = this.createMediaKeySessionContext(mediaKeySessionContext);
      const keyId = this.getKeyIdString(decryptdata);
      const scheme = "cenc";
      this.keyIdToKeySessionPromise[keyId] = this.generateRequestWithPreferredKeySession(keySessionContext, scheme, decryptdata.pssh, "expired");
    } else {
      this.warn(`Could not renew expired session. Missing pssh initData.`);
    }
    this.removeSession(mediaKeySessionContext);
  }
  getKeyIdString(decryptdata) {
    if (!decryptdata) {
      throw new Error("Could not read keyId of undefined decryptdata");
    }
    if (decryptdata.keyId === null) {
      throw new Error("keyId is null");
    }
    return Hex.hexDump(decryptdata.keyId);
  }
  updateKeySession(mediaKeySessionContext, data) {
    var _mediaKeySessionConte;
    const keySession = mediaKeySessionContext.mediaKeysSession;
    this.log(`Updating key-session "${keySession.sessionId}" for keyID ${Hex.hexDump(((_mediaKeySessionConte = mediaKeySessionContext.decryptdata) == null ? void 0 : _mediaKeySessionConte.keyId) || [])}
      } (data length: ${data ? data.byteLength : data})`);
    return keySession.update(data);
  }
  selectKeySystemFormat(frag) {
    const keyFormats = Object.keys(frag.levelkeys || {});
    if (!this.keyFormatPromise) {
      this.log(`Selecting key-system from fragment (sn: ${frag.sn} ${frag.type}: ${frag.level}) key formats ${keyFormats.join(", ")}`);
      this.keyFormatPromise = this.getKeyFormatPromise(keyFormats);
    }
    return this.keyFormatPromise;
  }
  getKeyFormatPromise(keyFormats) {
    return new Promise((resolve, reject) => {
      const keySystemsInConfig = getKeySystemsForConfig(this.config);
      const keySystemsToAttempt = keyFormats.map(keySystemFormatToKeySystemDomain).filter((value) => !!value && keySystemsInConfig.indexOf(value) !== -1);
      return this.getKeySystemSelectionPromise(keySystemsToAttempt).then(({
        keySystem
      }) => {
        const keySystemFormat = keySystemDomainToKeySystemFormat(keySystem);
        if (keySystemFormat) {
          resolve(keySystemFormat);
        } else {
          reject(new Error(`Unable to find format for key-system "${keySystem}"`));
        }
      }).catch(reject);
    });
  }
  loadKey(data) {
    const decryptdata = data.keyInfo.decryptdata;
    const keyId = this.getKeyIdString(decryptdata);
    const keyDetails = `(keyId: ${keyId} format: "${decryptdata.keyFormat}" method: ${decryptdata.method} uri: ${decryptdata.uri})`;
    this.log(`Starting session for key ${keyDetails}`);
    let keySessionContextPromise = this.keyIdToKeySessionPromise[keyId];
    if (!keySessionContextPromise) {
      keySessionContextPromise = this.keyIdToKeySessionPromise[keyId] = this.getKeySystemForKeyPromise(decryptdata).then(({
        keySystem,
        mediaKeys
      }) => {
        this.throwIfDestroyed();
        this.log(`Handle encrypted media sn: ${data.frag.sn} ${data.frag.type}: ${data.frag.level} using key ${keyDetails}`);
        return this.attemptSetMediaKeys(keySystem, mediaKeys).then(() => {
          this.throwIfDestroyed();
          const keySessionContext = this.createMediaKeySessionContext({
            keySystem,
            mediaKeys,
            decryptdata
          });
          const scheme = "cenc";
          return this.generateRequestWithPreferredKeySession(keySessionContext, scheme, decryptdata.pssh, "playlist-key");
        });
      });
      keySessionContextPromise.catch((error) => this.handleError(error));
    }
    return keySessionContextPromise;
  }
  throwIfDestroyed(message = "Invalid state") {
    if (!this.hls) {
      throw new Error("invalid state");
    }
  }
  handleError(error) {
    if (!this.hls) {
      return;
    }
    this.error(error.message);
    if (error instanceof EMEKeyError) {
      this.hls.trigger(Events.ERROR, error.data);
    } else {
      this.hls.trigger(Events.ERROR, {
        type: ErrorTypes.KEY_SYSTEM_ERROR,
        details: ErrorDetails.KEY_SYSTEM_NO_KEYS,
        error,
        fatal: true
      });
    }
  }
  getKeySystemForKeyPromise(decryptdata) {
    const keyId = this.getKeyIdString(decryptdata);
    const mediaKeySessionContext = this.keyIdToKeySessionPromise[keyId];
    if (!mediaKeySessionContext) {
      const keySystem = keySystemFormatToKeySystemDomain(decryptdata.keyFormat);
      const keySystemsToAttempt = keySystem ? [keySystem] : getKeySystemsForConfig(this.config);
      return this.attemptKeySystemAccess(keySystemsToAttempt);
    }
    return mediaKeySessionContext;
  }
  getKeySystemSelectionPromise(keySystemsToAttempt) {
    if (!keySystemsToAttempt.length) {
      keySystemsToAttempt = getKeySystemsForConfig(this.config);
    }
    if (keySystemsToAttempt.length === 0) {
      throw new EMEKeyError({
        type: ErrorTypes.KEY_SYSTEM_ERROR,
        details: ErrorDetails.KEY_SYSTEM_NO_CONFIGURED_LICENSE,
        fatal: true
      }, `Missing key-system license configuration options ${JSON.stringify({
        drmSystems: this.config.drmSystems
      })}`);
    }
    return this.attemptKeySystemAccess(keySystemsToAttempt);
  }
  _onMediaEncrypted(event) {
    const {
      initDataType,
      initData
    } = event;
    this.debug(`"${event.type}" event: init data type: "${initDataType}"`);
    if (initData === null) {
      return;
    }
    let keyId;
    let keySystemDomain;
    if (initDataType === "sinf" && this.config.drmSystems[KeySystems.FAIRPLAY]) {
      const json = bin2str(new Uint8Array(initData));
      try {
        const sinf = base64Decode(JSON.parse(json).sinf);
        const tenc = parseSinf(new Uint8Array(sinf));
        if (!tenc) {
          return;
        }
        keyId = tenc.subarray(8, 24);
        keySystemDomain = KeySystems.FAIRPLAY;
      } catch (error) {
        this.warn('Failed to parse sinf "encrypted" event message initData');
        return;
      }
    } else {
      const psshInfo = parsePssh(initData);
      if (psshInfo === null) {
        return;
      }
      if (psshInfo.version === 0 && psshInfo.systemId === KeySystemIds.WIDEVINE && psshInfo.data) {
        keyId = psshInfo.data.subarray(8, 24);
      }
      keySystemDomain = keySystemIdToKeySystemDomain(psshInfo.systemId);
    }
    if (!keySystemDomain || !keyId) {
      return;
    }
    const keyIdHex = Hex.hexDump(keyId);
    const {
      keyIdToKeySessionPromise,
      mediaKeySessions
    } = this;
    let keySessionContextPromise = keyIdToKeySessionPromise[keyIdHex];
    for (let i2 = 0; i2 < mediaKeySessions.length; i2++) {
      const keyContext = mediaKeySessions[i2];
      const decryptdata = keyContext.decryptdata;
      if (decryptdata.pssh || !decryptdata.keyId) {
        continue;
      }
      const oldKeyIdHex = Hex.hexDump(decryptdata.keyId);
      if (keyIdHex === oldKeyIdHex || decryptdata.uri.replace(/-/g, "").indexOf(keyIdHex) !== -1) {
        keySessionContextPromise = keyIdToKeySessionPromise[oldKeyIdHex];
        delete keyIdToKeySessionPromise[oldKeyIdHex];
        decryptdata.pssh = new Uint8Array(initData);
        decryptdata.keyId = keyId;
        keySessionContextPromise = keyIdToKeySessionPromise[keyIdHex] = keySessionContextPromise.then(() => {
          return this.generateRequestWithPreferredKeySession(keyContext, initDataType, initData, "encrypted-event-key-match");
        });
        break;
      }
    }
    if (!keySessionContextPromise) {
      keySessionContextPromise = keyIdToKeySessionPromise[keyIdHex] = this.getKeySystemSelectionPromise([keySystemDomain]).then(({
        keySystem,
        mediaKeys
      }) => {
        var _keySystemToKeySystem;
        this.throwIfDestroyed();
        const decryptdata = new LevelKey("ISO-23001-7", keyIdHex, (_keySystemToKeySystem = keySystemDomainToKeySystemFormat(keySystem)) != null ? _keySystemToKeySystem : "");
        decryptdata.pssh = new Uint8Array(initData);
        decryptdata.keyId = keyId;
        return this.attemptSetMediaKeys(keySystem, mediaKeys).then(() => {
          this.throwIfDestroyed();
          const keySessionContext = this.createMediaKeySessionContext({
            decryptdata,
            keySystem,
            mediaKeys
          });
          return this.generateRequestWithPreferredKeySession(keySessionContext, initDataType, initData, "encrypted-event-no-match");
        });
      });
    }
    keySessionContextPromise.catch((error) => this.handleError(error));
  }
  _onWaitingForKey(event) {
    this.log(`"${event.type}" event`);
  }
  attemptSetMediaKeys(keySystem, mediaKeys) {
    const queue = this.setMediaKeysQueue.slice();
    this.log(`Setting media-keys for "${keySystem}"`);
    const setMediaKeysPromise = Promise.all(queue).then(() => {
      if (!this.media) {
        throw new Error("Attempted to set mediaKeys without media element attached");
      }
      return this.media.setMediaKeys(mediaKeys);
    });
    this.setMediaKeysQueue.push(setMediaKeysPromise);
    return setMediaKeysPromise.then(() => {
      this.log(`Media-keys set for "${keySystem}"`);
      queue.push(setMediaKeysPromise);
      this.setMediaKeysQueue = this.setMediaKeysQueue.filter((p) => queue.indexOf(p) === -1);
    });
  }
  generateRequestWithPreferredKeySession(context, initDataType, initData, reason) {
    var _this$config$drmSyste, _this$config$drmSyste2;
    const generateRequestFilter = (_this$config$drmSyste = this.config.drmSystems) == null ? void 0 : (_this$config$drmSyste2 = _this$config$drmSyste[context.keySystem]) == null ? void 0 : _this$config$drmSyste2.generateRequest;
    if (generateRequestFilter) {
      try {
        const mappedInitData = generateRequestFilter.call(this.hls, initDataType, initData, context);
        if (!mappedInitData) {
          throw new Error("Invalid response from configured generateRequest filter");
        }
        initDataType = mappedInitData.initDataType;
        initData = context.decryptdata.pssh = mappedInitData.initData ? new Uint8Array(mappedInitData.initData) : null;
      } catch (error) {
        var _this$hls;
        this.warn(error.message);
        if ((_this$hls = this.hls) != null && _this$hls.config.debug) {
          throw error;
        }
      }
    }
    if (initData === null) {
      this.log(`Skipping key-session request for "${reason}" (no initData)`);
      return Promise.resolve(context);
    }
    const keyId = this.getKeyIdString(context.decryptdata);
    this.log(`Generating key-session request for "${reason}": ${keyId} (init data type: ${initDataType} length: ${initData ? initData.byteLength : null})`);
    const licenseStatus = new EventEmitter();
    const onmessage = context._onmessage = (event) => {
      const keySession = context.mediaKeysSession;
      if (!keySession) {
        licenseStatus.emit("error", new Error("invalid state"));
        return;
      }
      const {
        messageType,
        message
      } = event;
      this.log(`"${messageType}" message event for session "${keySession.sessionId}" message size: ${message.byteLength}`);
      if (messageType === "license-request" || messageType === "license-renewal") {
        this.renewLicense(context, message).catch((error) => {
          this.handleError(error);
          licenseStatus.emit("error", error);
        });
      } else if (messageType === "license-release") {
        if (context.keySystem === KeySystems.FAIRPLAY) {
          this.updateKeySession(context, strToUtf8array("acknowledged"));
          this.removeSession(context);
        }
      } else {
        this.warn(`unhandled media key message type "${messageType}"`);
      }
    };
    const onkeystatuseschange = context._onkeystatuseschange = (event) => {
      const keySession = context.mediaKeysSession;
      if (!keySession) {
        licenseStatus.emit("error", new Error("invalid state"));
        return;
      }
      this.onKeyStatusChange(context);
      const keyStatus = context.keyStatus;
      licenseStatus.emit("keyStatus", keyStatus);
      if (keyStatus === "expired") {
        this.warn(`${context.keySystem} expired for key ${keyId}`);
        this.renewKeySession(context);
      }
    };
    context.mediaKeysSession.addEventListener("message", onmessage);
    context.mediaKeysSession.addEventListener("keystatuseschange", onkeystatuseschange);
    const keyUsablePromise = new Promise((resolve, reject) => {
      licenseStatus.on("error", reject);
      licenseStatus.on("keyStatus", (keyStatus) => {
        if (keyStatus.startsWith("usable")) {
          resolve();
        } else if (keyStatus === "output-restricted") {
          reject(new EMEKeyError({
            type: ErrorTypes.KEY_SYSTEM_ERROR,
            details: ErrorDetails.KEY_SYSTEM_STATUS_OUTPUT_RESTRICTED,
            fatal: false
          }, "HDCP level output restricted"));
        } else if (keyStatus === "internal-error") {
          reject(new EMEKeyError({
            type: ErrorTypes.KEY_SYSTEM_ERROR,
            details: ErrorDetails.KEY_SYSTEM_STATUS_INTERNAL_ERROR,
            fatal: true
          }, `key status changed to "${keyStatus}"`));
        } else if (keyStatus === "expired") {
          reject(new Error("key expired while generating request"));
        } else {
          this.warn(`unhandled key status change "${keyStatus}"`);
        }
      });
    });
    return context.mediaKeysSession.generateRequest(initDataType, initData).then(() => {
      var _context$mediaKeysSes;
      this.log(`Request generated for key-session "${(_context$mediaKeysSes = context.mediaKeysSession) == null ? void 0 : _context$mediaKeysSes.sessionId}" keyId: ${keyId}`);
    }).catch((error) => {
      throw new EMEKeyError({
        type: ErrorTypes.KEY_SYSTEM_ERROR,
        details: ErrorDetails.KEY_SYSTEM_NO_SESSION,
        error,
        fatal: false
      }, `Error generating key-session request: ${error}`);
    }).then(() => keyUsablePromise).catch((error) => {
      licenseStatus.removeAllListeners();
      this.removeSession(context);
      throw error;
    }).then(() => {
      licenseStatus.removeAllListeners();
      return context;
    });
  }
  onKeyStatusChange(mediaKeySessionContext) {
    mediaKeySessionContext.mediaKeysSession.keyStatuses.forEach((status2, keyId) => {
      this.log(`key status change "${status2}" for keyStatuses keyId: ${Hex.hexDump("buffer" in keyId ? new Uint8Array(keyId.buffer, keyId.byteOffset, keyId.byteLength) : new Uint8Array(keyId))} session keyId: ${Hex.hexDump(new Uint8Array(mediaKeySessionContext.decryptdata.keyId || []))} uri: ${mediaKeySessionContext.decryptdata.uri}`);
      mediaKeySessionContext.keyStatus = status2;
    });
  }
  fetchServerCertificate(keySystem) {
    const config = this.config;
    const Loader = config.loader;
    const certLoader = new Loader(config);
    const url = this.getServerCertificateUrl(keySystem);
    if (!url) {
      return Promise.resolve();
    }
    this.log(`Fetching server certificate for "${keySystem}"`);
    return new Promise((resolve, reject) => {
      const loaderContext = {
        responseType: "arraybuffer",
        url
      };
      const loadPolicy = config.certLoadPolicy.default;
      const loaderConfig = {
        loadPolicy,
        timeout: loadPolicy.maxLoadTimeMs,
        maxRetry: 0,
        retryDelay: 0,
        maxRetryDelay: 0
      };
      const loaderCallbacks = {
        onSuccess: (response, stats, context, networkDetails) => {
          resolve(response.data);
        },
        onError: (response, contex, networkDetails, stats) => {
          reject(new EMEKeyError({
            type: ErrorTypes.KEY_SYSTEM_ERROR,
            details: ErrorDetails.KEY_SYSTEM_SERVER_CERTIFICATE_REQUEST_FAILED,
            fatal: true,
            networkDetails,
            response: _objectSpread2({
              url: loaderContext.url,
              data: void 0
            }, response)
          }, `"${keySystem}" certificate request failed (${url}). Status: ${response.code} (${response.text})`));
        },
        onTimeout: (stats, context, networkDetails) => {
          reject(new EMEKeyError({
            type: ErrorTypes.KEY_SYSTEM_ERROR,
            details: ErrorDetails.KEY_SYSTEM_SERVER_CERTIFICATE_REQUEST_FAILED,
            fatal: true,
            networkDetails,
            response: {
              url: loaderContext.url,
              data: void 0
            }
          }, `"${keySystem}" certificate request timed out (${url})`));
        },
        onAbort: (stats, context, networkDetails) => {
          reject(new Error("aborted"));
        }
      };
      certLoader.load(loaderContext, loaderConfig, loaderCallbacks);
    });
  }
  setMediaKeysServerCertificate(mediaKeys, keySystem, cert) {
    return new Promise((resolve, reject) => {
      mediaKeys.setServerCertificate(cert).then((success) => {
        this.log(`setServerCertificate ${success ? "success" : "not supported by CDM"} (${cert == null ? void 0 : cert.byteLength}) on "${keySystem}"`);
        resolve(mediaKeys);
      }).catch((error) => {
        reject(new EMEKeyError({
          type: ErrorTypes.KEY_SYSTEM_ERROR,
          details: ErrorDetails.KEY_SYSTEM_SERVER_CERTIFICATE_UPDATE_FAILED,
          error,
          fatal: true
        }, error.message));
      });
    });
  }
  renewLicense(context, keyMessage) {
    return this.requestLicense(context, new Uint8Array(keyMessage)).then((data) => {
      return this.updateKeySession(context, new Uint8Array(data)).catch((error) => {
        throw new EMEKeyError({
          type: ErrorTypes.KEY_SYSTEM_ERROR,
          details: ErrorDetails.KEY_SYSTEM_SESSION_UPDATE_FAILED,
          error,
          fatal: true
        }, error.message);
      });
    });
  }
  unpackPlayReadyKeyMessage(xhr, licenseChallenge) {
    const xmlString = String.fromCharCode.apply(null, new Uint16Array(licenseChallenge.buffer));
    if (!xmlString.includes("PlayReadyKeyMessage")) {
      xhr.setRequestHeader("Content-Type", "text/xml; charset=utf-8");
      return licenseChallenge;
    }
    const keyMessageXml = new DOMParser().parseFromString(xmlString, "application/xml");
    const headers = keyMessageXml.querySelectorAll("HttpHeader");
    if (headers.length > 0) {
      let header;
      for (let i2 = 0, len = headers.length; i2 < len; i2++) {
        var _header$querySelector, _header$querySelector2;
        header = headers[i2];
        const name = (_header$querySelector = header.querySelector("name")) == null ? void 0 : _header$querySelector.textContent;
        const value = (_header$querySelector2 = header.querySelector("value")) == null ? void 0 : _header$querySelector2.textContent;
        if (name && value) {
          xhr.setRequestHeader(name, value);
        }
      }
    }
    const challengeElement = keyMessageXml.querySelector("Challenge");
    const challengeText = challengeElement == null ? void 0 : challengeElement.textContent;
    if (!challengeText) {
      throw new Error(`Cannot find <Challenge> in key message`);
    }
    return strToUtf8array(atob(challengeText));
  }
  setupLicenseXHR(xhr, url, keysListItem, licenseChallenge) {
    const licenseXhrSetup = this.config.licenseXhrSetup;
    if (!licenseXhrSetup) {
      xhr.open("POST", url, true);
      return Promise.resolve({
        xhr,
        licenseChallenge
      });
    }
    return Promise.resolve().then(() => {
      if (!keysListItem.decryptdata) {
        throw new Error("Key removed");
      }
      return licenseXhrSetup.call(this.hls, xhr, url, keysListItem, licenseChallenge);
    }).catch((error) => {
      if (!keysListItem.decryptdata) {
        throw error;
      }
      xhr.open("POST", url, true);
      return licenseXhrSetup.call(this.hls, xhr, url, keysListItem, licenseChallenge);
    }).then((licenseXhrSetupResult) => {
      if (!xhr.readyState) {
        xhr.open("POST", url, true);
      }
      const finalLicenseChallenge = licenseXhrSetupResult ? licenseXhrSetupResult : licenseChallenge;
      return {
        xhr,
        licenseChallenge: finalLicenseChallenge
      };
    });
  }
  requestLicense(keySessionContext, licenseChallenge) {
    const keyLoadPolicy = this.config.keyLoadPolicy.default;
    return new Promise((resolve, reject) => {
      const url = this.getLicenseServerUrl(keySessionContext.keySystem);
      this.log(`Sending license request to URL: ${url}`);
      const xhr = new XMLHttpRequest();
      xhr.responseType = "arraybuffer";
      xhr.onreadystatechange = () => {
        if (!this.hls || !keySessionContext.mediaKeysSession) {
          return reject(new Error("invalid state"));
        }
        if (xhr.readyState === 4) {
          if (xhr.status === 200) {
            this._requestLicenseFailureCount = 0;
            let data = xhr.response;
            this.log(`License received ${data instanceof ArrayBuffer ? data.byteLength : data}`);
            const licenseResponseCallback = this.config.licenseResponseCallback;
            if (licenseResponseCallback) {
              try {
                data = licenseResponseCallback.call(this.hls, xhr, url, keySessionContext);
              } catch (error) {
                this.error(error);
              }
            }
            resolve(data);
          } else {
            const retryConfig = keyLoadPolicy.errorRetry;
            const maxNumRetry = retryConfig ? retryConfig.maxNumRetry : 0;
            this._requestLicenseFailureCount++;
            if (this._requestLicenseFailureCount > maxNumRetry || xhr.status >= 400 && xhr.status < 500) {
              reject(new EMEKeyError({
                type: ErrorTypes.KEY_SYSTEM_ERROR,
                details: ErrorDetails.KEY_SYSTEM_LICENSE_REQUEST_FAILED,
                fatal: true,
                networkDetails: xhr,
                response: {
                  url,
                  data: void 0,
                  code: xhr.status,
                  text: xhr.statusText
                }
              }, `License Request XHR failed (${url}). Status: ${xhr.status} (${xhr.statusText})`));
            } else {
              const attemptsLeft = maxNumRetry - this._requestLicenseFailureCount + 1;
              this.warn(`Retrying license request, ${attemptsLeft} attempts left`);
              this.requestLicense(keySessionContext, licenseChallenge).then(resolve, reject);
            }
          }
        }
      };
      if (keySessionContext.licenseXhr && keySessionContext.licenseXhr.readyState !== XMLHttpRequest.DONE) {
        keySessionContext.licenseXhr.abort();
      }
      keySessionContext.licenseXhr = xhr;
      this.setupLicenseXHR(xhr, url, keySessionContext, licenseChallenge).then(({
        xhr: xhr2,
        licenseChallenge: licenseChallenge2
      }) => {
        if (keySessionContext.keySystem == KeySystems.PLAYREADY) {
          licenseChallenge2 = this.unpackPlayReadyKeyMessage(xhr2, licenseChallenge2);
        }
        xhr2.send(licenseChallenge2);
      });
    });
  }
  onMediaAttached(event, data) {
    if (!this.config.emeEnabled) {
      return;
    }
    const media = data.media;
    this.media = media;
    media.addEventListener("encrypted", this.onMediaEncrypted);
    media.addEventListener("waitingforkey", this.onWaitingForKey);
  }
  onMediaDetached() {
    const media = this.media;
    const mediaKeysList = this.mediaKeySessions;
    if (media) {
      media.removeEventListener("encrypted", this.onMediaEncrypted);
      media.removeEventListener("waitingforkey", this.onWaitingForKey);
      this.media = null;
    }
    this._requestLicenseFailureCount = 0;
    this.setMediaKeysQueue = [];
    this.mediaKeySessions = [];
    this.keyIdToKeySessionPromise = {};
    LevelKey.clearKeyUriToKeyIdMap();
    const keySessionCount = mediaKeysList.length;
    EMEController.CDMCleanupPromise = Promise.all(mediaKeysList.map((mediaKeySessionContext) => this.removeSession(mediaKeySessionContext)).concat(media == null ? void 0 : media.setMediaKeys(null).catch((error) => {
      this.log(`Could not clear media keys: ${error}`);
    }))).then(() => {
      if (keySessionCount) {
        this.log("finished closing key sessions and clearing media keys");
        mediaKeysList.length = 0;
      }
    }).catch((error) => {
      this.log(`Could not close sessions and clear media keys: ${error}`);
    });
  }
  onManifestLoading() {
    this.keyFormatPromise = null;
  }
  onManifestLoaded(event, {
    sessionKeys
  }) {
    if (!sessionKeys || !this.config.emeEnabled) {
      return;
    }
    if (!this.keyFormatPromise) {
      const keyFormats = sessionKeys.reduce((formats, sessionKey) => {
        if (formats.indexOf(sessionKey.keyFormat) === -1) {
          formats.push(sessionKey.keyFormat);
        }
        return formats;
      }, []);
      this.log(`Selecting key-system from session-keys ${keyFormats.join(", ")}`);
      this.keyFormatPromise = this.getKeyFormatPromise(keyFormats);
    }
  }
  removeSession(mediaKeySessionContext) {
    const {
      mediaKeysSession,
      licenseXhr
    } = mediaKeySessionContext;
    if (mediaKeysSession) {
      this.log(`Remove licenses and keys and close session ${mediaKeysSession.sessionId}`);
      if (mediaKeySessionContext._onmessage) {
        mediaKeysSession.removeEventListener("message", mediaKeySessionContext._onmessage);
        mediaKeySessionContext._onmessage = void 0;
      }
      if (mediaKeySessionContext._onkeystatuseschange) {
        mediaKeysSession.removeEventListener("keystatuseschange", mediaKeySessionContext._onkeystatuseschange);
        mediaKeySessionContext._onkeystatuseschange = void 0;
      }
      if (licenseXhr && licenseXhr.readyState !== XMLHttpRequest.DONE) {
        licenseXhr.abort();
      }
      mediaKeySessionContext.mediaKeysSession = mediaKeySessionContext.decryptdata = mediaKeySessionContext.licenseXhr = void 0;
      const index = this.mediaKeySessions.indexOf(mediaKeySessionContext);
      if (index > -1) {
        this.mediaKeySessions.splice(index, 1);
      }
      return mediaKeysSession.remove().catch((error) => {
        this.log(`Could not remove session: ${error}`);
      }).then(() => {
        return mediaKeysSession.close();
      }).catch((error) => {
        this.log(`Could not close session: ${error}`);
      });
    }
  }
}
EMEController.CDMCleanupPromise = void 0;
class EMEKeyError extends Error {
  constructor(data, message) {
    super(message);
    this.data = void 0;
    data.error || (data.error = new Error(message));
    this.data = data;
    data.err = data.error;
  }
}
var CmObjectType;
(function(CmObjectType2) {
  CmObjectType2["MANIFEST"] = "m";
  CmObjectType2["AUDIO"] = "a";
  CmObjectType2["VIDEO"] = "v";
  CmObjectType2["MUXED"] = "av";
  CmObjectType2["INIT"] = "i";
  CmObjectType2["CAPTION"] = "c";
  CmObjectType2["TIMED_TEXT"] = "tt";
  CmObjectType2["KEY"] = "k";
  CmObjectType2["OTHER"] = "o";
})(CmObjectType || (CmObjectType = {}));
var CmStreamingFormat;
(function(CmStreamingFormat2) {
  CmStreamingFormat2["DASH"] = "d";
  CmStreamingFormat2["HLS"] = "h";
  CmStreamingFormat2["SMOOTH"] = "s";
  CmStreamingFormat2["OTHER"] = "o";
})(CmStreamingFormat || (CmStreamingFormat = {}));
var CmcdHeaderField;
(function(CmcdHeaderField2) {
  CmcdHeaderField2["OBJECT"] = "CMCD-Object";
  CmcdHeaderField2["REQUEST"] = "CMCD-Request";
  CmcdHeaderField2["SESSION"] = "CMCD-Session";
  CmcdHeaderField2["STATUS"] = "CMCD-Status";
})(CmcdHeaderField || (CmcdHeaderField = {}));
const CmcdHeaderMap = {
  [CmcdHeaderField.OBJECT]: ["br", "d", "ot", "tb"],
  [CmcdHeaderField.REQUEST]: ["bl", "dl", "mtp", "nor", "nrr", "su"],
  [CmcdHeaderField.SESSION]: ["cid", "pr", "sf", "sid", "st", "v"],
  [CmcdHeaderField.STATUS]: ["bs", "rtp"]
};
class SfItem {
  constructor(value, params) {
    this.value = void 0;
    this.params = void 0;
    if (Array.isArray(value)) {
      value = value.map((v) => v instanceof SfItem ? v : new SfItem(v));
    }
    this.value = value;
    this.params = params;
  }
}
class SfToken {
  constructor(description) {
    this.description = void 0;
    this.description = description;
  }
}
const DICT = "Dict";
function format(value) {
  if (Array.isArray(value)) {
    return JSON.stringify(value);
  }
  if (value instanceof Map) {
    return "Map{}";
  }
  if (value instanceof Set) {
    return "Set{}";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}
function throwError(action, src, type, cause) {
  return new Error(`failed to ${action} "${format(src)}" as ${type}`, {
    cause
  });
}
const BARE_ITEM = "Bare Item";
const BOOLEAN = "Boolean";
const BYTES = "Byte Sequence";
const DECIMAL = "Decimal";
const INTEGER = "Integer";
function isInvalidInt(value) {
  return value < -999999999999999 || 999999999999999 < value;
}
const STRING_REGEX = /[\x00-\x1f\x7f]+/;
const TOKEN = "Token";
const KEY = "Key";
function serializeError(src, type, cause) {
  return throwError("serialize", src, type, cause);
}
function serializeBoolean(value) {
  if (typeof value !== "boolean") {
    throw serializeError(value, BOOLEAN);
  }
  return value ? "?1" : "?0";
}
function base64encode(binary) {
  return btoa(String.fromCharCode(...binary));
}
function serializeByteSequence(value) {
  if (ArrayBuffer.isView(value) === false) {
    throw serializeError(value, BYTES);
  }
  return `:${base64encode(value)}:`;
}
function serializeInteger(value) {
  if (isInvalidInt(value)) {
    throw serializeError(value, INTEGER);
  }
  return value.toString();
}
function serializeDate(value) {
  return `@${serializeInteger(value.getTime() / 1e3)}`;
}
function roundToEven(value, precision) {
  if (value < 0) {
    return -roundToEven(-value, precision);
  }
  const decimalShift = Math.pow(10, precision);
  const isEquidistant = Math.abs(value * decimalShift % 1 - 0.5) < Number.EPSILON;
  if (isEquidistant) {
    const flooredValue = Math.floor(value * decimalShift);
    return (flooredValue % 2 === 0 ? flooredValue : flooredValue + 1) / decimalShift;
  } else {
    return Math.round(value * decimalShift) / decimalShift;
  }
}
function serializeDecimal(value) {
  const roundedValue = roundToEven(value, 3);
  if (Math.floor(Math.abs(roundedValue)).toString().length > 12) {
    throw serializeError(value, DECIMAL);
  }
  const stringValue = roundedValue.toString();
  return stringValue.includes(".") ? stringValue : `${stringValue}.0`;
}
const STRING = "String";
function serializeString(value) {
  if (STRING_REGEX.test(value)) {
    throw serializeError(value, STRING);
  }
  return `"${value.replace(/\\/g, `\\\\`).replace(/"/g, `\\"`)}"`;
}
function symbolToStr(symbol) {
  return symbol.description || symbol.toString().slice(7, -1);
}
function serializeToken(token) {
  const value = symbolToStr(token);
  if (/^([a-zA-Z*])([!#$%&'*+\-.^_`|~\w:/]*)$/.test(value) === false) {
    throw serializeError(value, TOKEN);
  }
  return value;
}
function serializeBareItem(value) {
  switch (typeof value) {
    case "number":
      if (!isFiniteNumber(value)) {
        throw serializeError(value, BARE_ITEM);
      }
      if (Number.isInteger(value)) {
        return serializeInteger(value);
      }
      return serializeDecimal(value);
    case "string":
      return serializeString(value);
    case "symbol":
      return serializeToken(value);
    case "boolean":
      return serializeBoolean(value);
    case "object":
      if (value instanceof Date) {
        return serializeDate(value);
      }
      if (value instanceof Uint8Array) {
        return serializeByteSequence(value);
      }
      if (value instanceof SfToken) {
        return serializeToken(value);
      }
    default:
      throw serializeError(value, BARE_ITEM);
  }
}
function serializeKey(value) {
  if (/^[a-z*][a-z0-9\-_.*]*$/.test(value) === false) {
    throw serializeError(value, KEY);
  }
  return value;
}
function serializeParams(params) {
  if (params == null) {
    return "";
  }
  return Object.entries(params).map(([key, value]) => {
    if (value === true) {
      return `;${serializeKey(key)}`;
    }
    return `;${serializeKey(key)}=${serializeBareItem(value)}`;
  }).join("");
}
function serializeItem(value) {
  if (value instanceof SfItem) {
    return `${serializeBareItem(value.value)}${serializeParams(value.params)}`;
  } else {
    return serializeBareItem(value);
  }
}
function serializeInnerList(value) {
  return `(${value.value.map(serializeItem).join(" ")})${serializeParams(value.params)}`;
}
function serializeDict(dict, options = {
  whitespace: true
}) {
  if (typeof dict !== "object") {
    throw serializeError(dict, DICT);
  }
  const entries = dict instanceof Map ? dict.entries() : Object.entries(dict);
  const optionalWhiteSpace = options != null && options.whitespace ? " " : "";
  return Array.from(entries).map(([key, item]) => {
    if (item instanceof SfItem === false) {
      item = new SfItem(item);
    }
    let output = serializeKey(key);
    if (item.value === true) {
      output += serializeParams(item.params);
    } else {
      output += "=";
      if (Array.isArray(item.value)) {
        output += serializeInnerList(item);
      } else {
        output += serializeItem(item);
      }
    }
    return output;
  }).join(`,${optionalWhiteSpace}`);
}
function encodeSfDict(value, options) {
  return serializeDict(value, options);
}
const isTokenField = (key) => key === "ot" || key === "sf" || key === "st";
const isValid = (value) => {
  if (typeof value === "number") {
    return isFiniteNumber(value);
  }
  return value != null && value !== "" && value !== false;
};
function urlToRelativePath(url, base) {
  const to = new URL(url);
  const from = new URL(base);
  if (to.origin !== from.origin) {
    return url;
  }
  const toPath = to.pathname.split("/").slice(1);
  const fromPath = from.pathname.split("/").slice(1, -1);
  while (toPath[0] === fromPath[0]) {
    toPath.shift();
    fromPath.shift();
  }
  while (fromPath.length) {
    fromPath.shift();
    toPath.unshift("..");
  }
  return toPath.join("/");
}
function uuid() {
  try {
    return crypto.randomUUID();
  } catch (error) {
    try {
      const url = URL.createObjectURL(new Blob());
      const uuid2 = url.toString();
      URL.revokeObjectURL(url);
      return uuid2.slice(uuid2.lastIndexOf("/") + 1);
    } catch (error2) {
      let dt = (/* @__PURE__ */ new Date()).getTime();
      const uuid2 = "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
        const r2 = (dt + Math.random() * 16) % 16 | 0;
        dt = Math.floor(dt / 16);
        return (c == "x" ? r2 : r2 & 3 | 8).toString(16);
      });
      return uuid2;
    }
  }
}
const toRounded = (value) => Math.round(value);
const toUrlSafe = (value, options) => {
  if (options != null && options.baseUrl) {
    value = urlToRelativePath(value, options.baseUrl);
  }
  return encodeURIComponent(value);
};
const toHundred = (value) => toRounded(value / 100) * 100;
const CmcdFormatters = {
  /**
   * Bitrate (kbps) rounded integer
   */
  br: toRounded,
  /**
   * Duration (milliseconds) rounded integer
   */
  d: toRounded,
  /**
   * Buffer Length (milliseconds) rounded nearest 100ms
   */
  bl: toHundred,
  /**
   * Deadline (milliseconds) rounded nearest 100ms
   */
  dl: toHundred,
  /**
   * Measured Throughput (kbps) rounded nearest 100kbps
   */
  mtp: toHundred,
  /**
   * Next Object Request URL encoded
   */
  nor: toUrlSafe,
  /**
   * Requested maximum throughput (kbps) rounded nearest 100kbps
   */
  rtp: toHundred,
  /**
   * Top Bitrate (kbps) rounded integer
   */
  tb: toRounded
};
function processCmcd(obj, options) {
  const results = {};
  if (obj == null || typeof obj !== "object") {
    return results;
  }
  const keys = Object.keys(obj).sort();
  const formatters = _extends({}, CmcdFormatters, options == null ? void 0 : options.formatters);
  const filter = options == null ? void 0 : options.filter;
  keys.forEach((key) => {
    if (filter != null && filter(key)) {
      return;
    }
    let value = obj[key];
    const formatter = formatters[key];
    if (formatter) {
      value = formatter(value, options);
    }
    if (key === "v" && value === 1) {
      return;
    }
    if (key == "pr" && value === 1) {
      return;
    }
    if (!isValid(value)) {
      return;
    }
    if (isTokenField(key) && typeof value === "string") {
      value = new SfToken(value);
    }
    results[key] = value;
  });
  return results;
}
function encodeCmcd(cmcd, options = {}) {
  if (!cmcd) {
    return "";
  }
  return encodeSfDict(processCmcd(cmcd, options), _extends({
    whitespace: false
  }, options));
}
function toCmcdHeaders(cmcd, options = {}) {
  if (!cmcd) {
    return {};
  }
  const entries = Object.entries(cmcd);
  const headerMap = Object.entries(CmcdHeaderMap).concat(Object.entries((options == null ? void 0 : options.customHeaderMap) || {}));
  const shards = entries.reduce((acc, entry) => {
    var _headerMap$find, _acc$field;
    const [key, value] = entry;
    const field = ((_headerMap$find = headerMap.find((entry2) => entry2[1].includes(key))) == null ? void 0 : _headerMap$find[0]) || CmcdHeaderField.REQUEST;
    (_acc$field = acc[field]) != null ? _acc$field : acc[field] = {};
    acc[field][key] = value;
    return acc;
  }, {});
  return Object.entries(shards).reduce((acc, [field, value]) => {
    acc[field] = encodeCmcd(value, options);
    return acc;
  }, {});
}
function appendCmcdHeaders(headers, cmcd, options) {
  return _extends(headers, toCmcdHeaders(cmcd, options));
}
const CMCD_PARAM = "CMCD";
function toCmcdQuery(cmcd, options = {}) {
  if (!cmcd) {
    return "";
  }
  const params = encodeCmcd(cmcd, options);
  return `${CMCD_PARAM}=${encodeURIComponent(params)}`;
}
const REGEX = /CMCD=[^&#]+/;
function appendCmcdQuery(url, cmcd, options) {
  const query = toCmcdQuery(cmcd, options);
  if (!query) {
    return url;
  }
  if (REGEX.test(url)) {
    return url.replace(REGEX, query);
  }
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}${query}`;
}
class CMCDController {
  // eslint-disable-line no-restricted-globals
  constructor(hls) {
    this.hls = void 0;
    this.config = void 0;
    this.media = void 0;
    this.sid = void 0;
    this.cid = void 0;
    this.useHeaders = false;
    this.includeKeys = void 0;
    this.initialized = false;
    this.starved = false;
    this.buffering = true;
    this.audioBuffer = void 0;
    this.videoBuffer = void 0;
    this.onWaiting = () => {
      if (this.initialized) {
        this.starved = true;
      }
      this.buffering = true;
    };
    this.onPlaying = () => {
      if (!this.initialized) {
        this.initialized = true;
      }
      this.buffering = false;
    };
    this.applyPlaylistData = (context) => {
      try {
        this.apply(context, {
          ot: CmObjectType.MANIFEST,
          su: !this.initialized
        });
      } catch (error) {
        logger.warn("Could not generate manifest CMCD data.", error);
      }
    };
    this.applyFragmentData = (context) => {
      try {
        const fragment = context.frag;
        const level = this.hls.levels[fragment.level];
        const ot = this.getObjectType(fragment);
        const data = {
          d: fragment.duration * 1e3,
          ot
        };
        if (ot === CmObjectType.VIDEO || ot === CmObjectType.AUDIO || ot == CmObjectType.MUXED) {
          data.br = level.bitrate / 1e3;
          data.tb = this.getTopBandwidth(ot) / 1e3;
          data.bl = this.getBufferLength(ot);
        }
        this.apply(context, data);
      } catch (error) {
        logger.warn("Could not generate segment CMCD data.", error);
      }
    };
    this.hls = hls;
    const config = this.config = hls.config;
    const {
      cmcd
    } = config;
    if (cmcd != null) {
      config.pLoader = this.createPlaylistLoader();
      config.fLoader = this.createFragmentLoader();
      this.sid = cmcd.sessionId || uuid();
      this.cid = cmcd.contentId;
      this.useHeaders = cmcd.useHeaders === true;
      this.includeKeys = cmcd.includeKeys;
      this.registerListeners();
    }
  }
  registerListeners() {
    const hls = this.hls;
    hls.on(Events.MEDIA_ATTACHED, this.onMediaAttached, this);
    hls.on(Events.MEDIA_DETACHED, this.onMediaDetached, this);
    hls.on(Events.BUFFER_CREATED, this.onBufferCreated, this);
  }
  unregisterListeners() {
    const hls = this.hls;
    hls.off(Events.MEDIA_ATTACHED, this.onMediaAttached, this);
    hls.off(Events.MEDIA_DETACHED, this.onMediaDetached, this);
    hls.off(Events.BUFFER_CREATED, this.onBufferCreated, this);
  }
  destroy() {
    this.unregisterListeners();
    this.onMediaDetached();
    this.hls = this.config = this.audioBuffer = this.videoBuffer = null;
    this.onWaiting = this.onPlaying = null;
  }
  onMediaAttached(event, data) {
    this.media = data.media;
    this.media.addEventListener("waiting", this.onWaiting);
    this.media.addEventListener("playing", this.onPlaying);
  }
  onMediaDetached() {
    if (!this.media) {
      return;
    }
    this.media.removeEventListener("waiting", this.onWaiting);
    this.media.removeEventListener("playing", this.onPlaying);
    this.media = null;
  }
  onBufferCreated(event, data) {
    var _data$tracks$audio, _data$tracks$video;
    this.audioBuffer = (_data$tracks$audio = data.tracks.audio) == null ? void 0 : _data$tracks$audio.buffer;
    this.videoBuffer = (_data$tracks$video = data.tracks.video) == null ? void 0 : _data$tracks$video.buffer;
  }
  /**
   * Create baseline CMCD data
   */
  createData() {
    var _this$media;
    return {
      v: 1,
      sf: CmStreamingFormat.HLS,
      sid: this.sid,
      cid: this.cid,
      pr: (_this$media = this.media) == null ? void 0 : _this$media.playbackRate,
      mtp: this.hls.bandwidthEstimate / 1e3
    };
  }
  /**
   * Apply CMCD data to a request.
   */
  apply(context, data = {}) {
    _extends(data, this.createData());
    const isVideo = data.ot === CmObjectType.INIT || data.ot === CmObjectType.VIDEO || data.ot === CmObjectType.MUXED;
    if (this.starved && isVideo) {
      data.bs = true;
      data.su = true;
      this.starved = false;
    }
    if (data.su == null) {
      data.su = this.buffering;
    }
    const {
      includeKeys
    } = this;
    if (includeKeys) {
      data = Object.keys(data).reduce((acc, key) => {
        includeKeys.includes(key) && (acc[key] = data[key]);
        return acc;
      }, {});
    }
    if (this.useHeaders) {
      if (!context.headers) {
        context.headers = {};
      }
      appendCmcdHeaders(context.headers, data);
    } else {
      context.url = appendCmcdQuery(context.url, data);
    }
  }
  /**
   * The CMCD object type.
   */
  getObjectType(fragment) {
    const {
      type
    } = fragment;
    if (type === "subtitle") {
      return CmObjectType.TIMED_TEXT;
    }
    if (fragment.sn === "initSegment") {
      return CmObjectType.INIT;
    }
    if (type === "audio") {
      return CmObjectType.AUDIO;
    }
    if (type === "main") {
      if (!this.hls.audioTracks.length) {
        return CmObjectType.MUXED;
      }
      return CmObjectType.VIDEO;
    }
    return void 0;
  }
  /**
   * Get the highest bitrate.
   */
  getTopBandwidth(type) {
    let bitrate = 0;
    let levels;
    const hls = this.hls;
    if (type === CmObjectType.AUDIO) {
      levels = hls.audioTracks;
    } else {
      const max = hls.maxAutoLevel;
      const len = max > -1 ? max + 1 : hls.levels.length;
      levels = hls.levels.slice(0, len);
    }
    for (const level of levels) {
      if (level.bitrate > bitrate) {
        bitrate = level.bitrate;
      }
    }
    return bitrate > 0 ? bitrate : NaN;
  }
  /**
   * Get the buffer length for a media type in milliseconds
   */
  getBufferLength(type) {
    const media = this.hls.media;
    const buffer = type === CmObjectType.AUDIO ? this.audioBuffer : this.videoBuffer;
    if (!buffer || !media) {
      return NaN;
    }
    const info = BufferHelper.bufferInfo(buffer, media.currentTime, this.config.maxBufferHole);
    return info.len * 1e3;
  }
  /**
   * Create a playlist loader
   */
  createPlaylistLoader() {
    const {
      pLoader
    } = this.config;
    const apply = this.applyPlaylistData;
    const Ctor = pLoader || this.config.loader;
    return class CmcdPlaylistLoader {
      constructor(config) {
        this.loader = void 0;
        this.loader = new Ctor(config);
      }
      get stats() {
        return this.loader.stats;
      }
      get context() {
        return this.loader.context;
      }
      destroy() {
        this.loader.destroy();
      }
      abort() {
        this.loader.abort();
      }
      load(context, config, callbacks) {
        apply(context);
        this.loader.load(context, config, callbacks);
      }
    };
  }
  /**
   * Create a playlist loader
   */
  createFragmentLoader() {
    const {
      fLoader
    } = this.config;
    const apply = this.applyFragmentData;
    const Ctor = fLoader || this.config.loader;
    return class CmcdFragmentLoader {
      constructor(config) {
        this.loader = void 0;
        this.loader = new Ctor(config);
      }
      get stats() {
        return this.loader.stats;
      }
      get context() {
        return this.loader.context;
      }
      destroy() {
        this.loader.destroy();
      }
      abort() {
        this.loader.abort();
      }
      load(context, config, callbacks) {
        apply(context);
        this.loader.load(context, config, callbacks);
      }
    };
  }
}
const PATHWAY_PENALTY_DURATION_MS = 3e5;
class ContentSteeringController {
  constructor(hls) {
    this.hls = void 0;
    this.log = void 0;
    this.loader = null;
    this.uri = null;
    this.pathwayId = ".";
    this.pathwayPriority = null;
    this.timeToLoad = 300;
    this.reloadTimer = -1;
    this.updated = 0;
    this.started = false;
    this.enabled = true;
    this.levels = null;
    this.audioTracks = null;
    this.subtitleTracks = null;
    this.penalizedPathways = {};
    this.hls = hls;
    this.log = logger.log.bind(logger, `[content-steering]:`);
    this.registerListeners();
  }
  registerListeners() {
    const hls = this.hls;
    hls.on(Events.MANIFEST_LOADING, this.onManifestLoading, this);
    hls.on(Events.MANIFEST_LOADED, this.onManifestLoaded, this);
    hls.on(Events.MANIFEST_PARSED, this.onManifestParsed, this);
    hls.on(Events.ERROR, this.onError, this);
  }
  unregisterListeners() {
    const hls = this.hls;
    if (!hls) {
      return;
    }
    hls.off(Events.MANIFEST_LOADING, this.onManifestLoading, this);
    hls.off(Events.MANIFEST_LOADED, this.onManifestLoaded, this);
    hls.off(Events.MANIFEST_PARSED, this.onManifestParsed, this);
    hls.off(Events.ERROR, this.onError, this);
  }
  startLoad() {
    this.started = true;
    this.clearTimeout();
    if (this.enabled && this.uri) {
      if (this.updated) {
        const ttl = this.timeToLoad * 1e3 - (performance.now() - this.updated);
        if (ttl > 0) {
          this.scheduleRefresh(this.uri, ttl);
          return;
        }
      }
      this.loadSteeringManifest(this.uri);
    }
  }
  stopLoad() {
    this.started = false;
    if (this.loader) {
      this.loader.destroy();
      this.loader = null;
    }
    this.clearTimeout();
  }
  clearTimeout() {
    if (this.reloadTimer !== -1) {
      self.clearTimeout(this.reloadTimer);
      this.reloadTimer = -1;
    }
  }
  destroy() {
    this.unregisterListeners();
    this.stopLoad();
    this.hls = null;
    this.levels = this.audioTracks = this.subtitleTracks = null;
  }
  removeLevel(levelToRemove) {
    const levels = this.levels;
    if (levels) {
      this.levels = levels.filter((level) => level !== levelToRemove);
    }
  }
  onManifestLoading() {
    this.stopLoad();
    this.enabled = true;
    this.timeToLoad = 300;
    this.updated = 0;
    this.uri = null;
    this.pathwayId = ".";
    this.levels = this.audioTracks = this.subtitleTracks = null;
  }
  onManifestLoaded(event, data) {
    const {
      contentSteering
    } = data;
    if (contentSteering === null) {
      return;
    }
    this.pathwayId = contentSteering.pathwayId;
    this.uri = contentSteering.uri;
    if (this.started) {
      this.startLoad();
    }
  }
  onManifestParsed(event, data) {
    this.audioTracks = data.audioTracks;
    this.subtitleTracks = data.subtitleTracks;
  }
  onError(event, data) {
    const {
      errorAction
    } = data;
    if ((errorAction == null ? void 0 : errorAction.action) === NetworkErrorAction.SendAlternateToPenaltyBox && errorAction.flags === ErrorActionFlags.MoveAllAlternatesMatchingHost) {
      const levels = this.levels;
      let pathwayPriority = this.pathwayPriority;
      let errorPathway = this.pathwayId;
      if (data.context) {
        const {
          groupId,
          pathwayId,
          type
        } = data.context;
        if (groupId && levels) {
          errorPathway = this.getPathwayForGroupId(groupId, type, errorPathway);
        } else if (pathwayId) {
          errorPathway = pathwayId;
        }
      }
      if (!(errorPathway in this.penalizedPathways)) {
        this.penalizedPathways[errorPathway] = performance.now();
      }
      if (!pathwayPriority && levels) {
        pathwayPriority = levels.reduce((pathways, level) => {
          if (pathways.indexOf(level.pathwayId) === -1) {
            pathways.push(level.pathwayId);
          }
          return pathways;
        }, []);
      }
      if (pathwayPriority && pathwayPriority.length > 1) {
        this.updatePathwayPriority(pathwayPriority);
        errorAction.resolved = this.pathwayId !== errorPathway;
      }
      if (!errorAction.resolved) {
        logger.warn(`Could not resolve ${data.details} ("${data.error.message}") with content-steering for Pathway: ${errorPathway} levels: ${levels ? levels.length : levels} priorities: ${JSON.stringify(pathwayPriority)} penalized: ${JSON.stringify(this.penalizedPathways)}`);
      }
    }
  }
  filterParsedLevels(levels) {
    this.levels = levels;
    let pathwayLevels = this.getLevelsForPathway(this.pathwayId);
    if (pathwayLevels.length === 0) {
      const pathwayId = levels[0].pathwayId;
      this.log(`No levels found in Pathway ${this.pathwayId}. Setting initial Pathway to "${pathwayId}"`);
      pathwayLevels = this.getLevelsForPathway(pathwayId);
      this.pathwayId = pathwayId;
    }
    if (pathwayLevels.length !== levels.length) {
      this.log(`Found ${pathwayLevels.length}/${levels.length} levels in Pathway "${this.pathwayId}"`);
      return pathwayLevels;
    }
    return levels;
  }
  getLevelsForPathway(pathwayId) {
    if (this.levels === null) {
      return [];
    }
    return this.levels.filter((level) => pathwayId === level.pathwayId);
  }
  updatePathwayPriority(pathwayPriority) {
    this.pathwayPriority = pathwayPriority;
    let levels;
    const penalizedPathways = this.penalizedPathways;
    const now2 = performance.now();
    Object.keys(penalizedPathways).forEach((pathwayId) => {
      if (now2 - penalizedPathways[pathwayId] > PATHWAY_PENALTY_DURATION_MS) {
        delete penalizedPathways[pathwayId];
      }
    });
    for (let i2 = 0; i2 < pathwayPriority.length; i2++) {
      const pathwayId = pathwayPriority[i2];
      if (pathwayId in penalizedPathways) {
        continue;
      }
      if (pathwayId === this.pathwayId) {
        return;
      }
      const selectedIndex = this.hls.nextLoadLevel;
      const selectedLevel = this.hls.levels[selectedIndex];
      levels = this.getLevelsForPathway(pathwayId);
      if (levels.length > 0) {
        this.log(`Setting Pathway to "${pathwayId}"`);
        this.pathwayId = pathwayId;
        reassignFragmentLevelIndexes(levels);
        this.hls.trigger(Events.LEVELS_UPDATED, {
          levels
        });
        const levelAfterChange = this.hls.levels[selectedIndex];
        if (selectedLevel && levelAfterChange && this.levels) {
          if (levelAfterChange.attrs["STABLE-VARIANT-ID"] !== selectedLevel.attrs["STABLE-VARIANT-ID"] && levelAfterChange.bitrate !== selectedLevel.bitrate) {
            this.log(`Unstable Pathways change from bitrate ${selectedLevel.bitrate} to ${levelAfterChange.bitrate}`);
          }
          this.hls.nextLoadLevel = selectedIndex;
        }
        break;
      }
    }
  }
  getPathwayForGroupId(groupId, type, defaultPathway) {
    const levels = this.getLevelsForPathway(defaultPathway).concat(this.levels || []);
    for (let i2 = 0; i2 < levels.length; i2++) {
      if (type === PlaylistContextType.AUDIO_TRACK && levels[i2].hasAudioGroup(groupId) || type === PlaylistContextType.SUBTITLE_TRACK && levels[i2].hasSubtitleGroup(groupId)) {
        return levels[i2].pathwayId;
      }
    }
    return defaultPathway;
  }
  clonePathways(pathwayClones) {
    const levels = this.levels;
    if (!levels) {
      return;
    }
    const audioGroupCloneMap = {};
    const subtitleGroupCloneMap = {};
    pathwayClones.forEach((pathwayClone) => {
      const {
        ID: cloneId,
        "BASE-ID": baseId,
        "URI-REPLACEMENT": uriReplacement
      } = pathwayClone;
      if (levels.some((level) => level.pathwayId === cloneId)) {
        return;
      }
      const clonedVariants = this.getLevelsForPathway(baseId).map((baseLevel) => {
        const attributes = new AttrList(baseLevel.attrs);
        attributes["PATHWAY-ID"] = cloneId;
        const clonedAudioGroupId = attributes.AUDIO && `${attributes.AUDIO}_clone_${cloneId}`;
        const clonedSubtitleGroupId = attributes.SUBTITLES && `${attributes.SUBTITLES}_clone_${cloneId}`;
        if (clonedAudioGroupId) {
          audioGroupCloneMap[attributes.AUDIO] = clonedAudioGroupId;
          attributes.AUDIO = clonedAudioGroupId;
        }
        if (clonedSubtitleGroupId) {
          subtitleGroupCloneMap[attributes.SUBTITLES] = clonedSubtitleGroupId;
          attributes.SUBTITLES = clonedSubtitleGroupId;
        }
        const url = performUriReplacement(baseLevel.uri, attributes["STABLE-VARIANT-ID"], "PER-VARIANT-URIS", uriReplacement);
        const clonedLevel = new Level({
          attrs: attributes,
          audioCodec: baseLevel.audioCodec,
          bitrate: baseLevel.bitrate,
          height: baseLevel.height,
          name: baseLevel.name,
          url,
          videoCodec: baseLevel.videoCodec,
          width: baseLevel.width
        });
        if (baseLevel.audioGroups) {
          for (let i2 = 1; i2 < baseLevel.audioGroups.length; i2++) {
            clonedLevel.addGroupId("audio", `${baseLevel.audioGroups[i2]}_clone_${cloneId}`);
          }
        }
        if (baseLevel.subtitleGroups) {
          for (let i2 = 1; i2 < baseLevel.subtitleGroups.length; i2++) {
            clonedLevel.addGroupId("text", `${baseLevel.subtitleGroups[i2]}_clone_${cloneId}`);
          }
        }
        return clonedLevel;
      });
      levels.push(...clonedVariants);
      cloneRenditionGroups(this.audioTracks, audioGroupCloneMap, uriReplacement, cloneId);
      cloneRenditionGroups(this.subtitleTracks, subtitleGroupCloneMap, uriReplacement, cloneId);
    });
  }
  loadSteeringManifest(uri) {
    const config = this.hls.config;
    const Loader = config.loader;
    if (this.loader) {
      this.loader.destroy();
    }
    this.loader = new Loader(config);
    let url;
    try {
      url = new self.URL(uri);
    } catch (error) {
      this.enabled = false;
      this.log(`Failed to parse Steering Manifest URI: ${uri}`);
      return;
    }
    if (url.protocol !== "data:") {
      const throughput = (this.hls.bandwidthEstimate || config.abrEwmaDefaultEstimate) | 0;
      url.searchParams.set("_HLS_pathway", this.pathwayId);
      url.searchParams.set("_HLS_throughput", "" + throughput);
    }
    const context = {
      responseType: "json",
      url: url.href
    };
    const loadPolicy = config.steeringManifestLoadPolicy.default;
    const legacyRetryCompatibility = loadPolicy.errorRetry || loadPolicy.timeoutRetry || {};
    const loaderConfig = {
      loadPolicy,
      timeout: loadPolicy.maxLoadTimeMs,
      maxRetry: legacyRetryCompatibility.maxNumRetry || 0,
      retryDelay: legacyRetryCompatibility.retryDelayMs || 0,
      maxRetryDelay: legacyRetryCompatibility.maxRetryDelayMs || 0
    };
    const callbacks = {
      onSuccess: (response, stats, context2, networkDetails) => {
        this.log(`Loaded steering manifest: "${url}"`);
        const steeringData = response.data;
        if (steeringData.VERSION !== 1) {
          this.log(`Steering VERSION ${steeringData.VERSION} not supported!`);
          return;
        }
        this.updated = performance.now();
        this.timeToLoad = steeringData.TTL;
        const {
          "RELOAD-URI": reloadUri,
          "PATHWAY-CLONES": pathwayClones,
          "PATHWAY-PRIORITY": pathwayPriority
        } = steeringData;
        if (reloadUri) {
          try {
            this.uri = new self.URL(reloadUri, url).href;
          } catch (error) {
            this.enabled = false;
            this.log(`Failed to parse Steering Manifest RELOAD-URI: ${reloadUri}`);
            return;
          }
        }
        this.scheduleRefresh(this.uri || context2.url);
        if (pathwayClones) {
          this.clonePathways(pathwayClones);
        }
        const loadedSteeringData = {
          steeringManifest: steeringData,
          url: url.toString()
        };
        this.hls.trigger(Events.STEERING_MANIFEST_LOADED, loadedSteeringData);
        if (pathwayPriority) {
          this.updatePathwayPriority(pathwayPriority);
        }
      },
      onError: (error, context2, networkDetails, stats) => {
        this.log(`Error loading steering manifest: ${error.code} ${error.text} (${context2.url})`);
        this.stopLoad();
        if (error.code === 410) {
          this.enabled = false;
          this.log(`Steering manifest ${context2.url} no longer available`);
          return;
        }
        let ttl = this.timeToLoad * 1e3;
        if (error.code === 429) {
          const loader = this.loader;
          if (typeof (loader == null ? void 0 : loader.getResponseHeader) === "function") {
            const retryAfter = loader.getResponseHeader("Retry-After");
            if (retryAfter) {
              ttl = parseFloat(retryAfter) * 1e3;
            }
          }
          this.log(`Steering manifest ${context2.url} rate limited`);
          return;
        }
        this.scheduleRefresh(this.uri || context2.url, ttl);
      },
      onTimeout: (stats, context2, networkDetails) => {
        this.log(`Timeout loading steering manifest (${context2.url})`);
        this.scheduleRefresh(this.uri || context2.url);
      }
    };
    this.log(`Requesting steering manifest: ${url}`);
    this.loader.load(context, loaderConfig, callbacks);
  }
  scheduleRefresh(uri, ttlMs = this.timeToLoad * 1e3) {
    this.clearTimeout();
    this.reloadTimer = self.setTimeout(() => {
      var _this$hls;
      const media = (_this$hls = this.hls) == null ? void 0 : _this$hls.media;
      if (media && !media.ended) {
        this.loadSteeringManifest(uri);
        return;
      }
      this.scheduleRefresh(uri, this.timeToLoad * 1e3);
    }, ttlMs);
  }
}
function cloneRenditionGroups(tracks, groupCloneMap, uriReplacement, cloneId) {
  if (!tracks) {
    return;
  }
  Object.keys(groupCloneMap).forEach((audioGroupId) => {
    const clonedTracks = tracks.filter((track) => track.groupId === audioGroupId).map((track) => {
      const clonedTrack = _extends({}, track);
      clonedTrack.details = void 0;
      clonedTrack.attrs = new AttrList(clonedTrack.attrs);
      clonedTrack.url = clonedTrack.attrs.URI = performUriReplacement(track.url, track.attrs["STABLE-RENDITION-ID"], "PER-RENDITION-URIS", uriReplacement);
      clonedTrack.groupId = clonedTrack.attrs["GROUP-ID"] = groupCloneMap[audioGroupId];
      clonedTrack.attrs["PATHWAY-ID"] = cloneId;
      return clonedTrack;
    });
    tracks.push(...clonedTracks);
  });
}
function performUriReplacement(uri, stableId, perOptionKey, uriReplacement) {
  const {
    HOST: host,
    PARAMS: params,
    [perOptionKey]: perOptionUris
  } = uriReplacement;
  let perVariantUri;
  if (stableId) {
    perVariantUri = perOptionUris == null ? void 0 : perOptionUris[stableId];
    if (perVariantUri) {
      uri = perVariantUri;
    }
  }
  const url = new self.URL(uri);
  if (host && !perVariantUri) {
    url.host = host;
  }
  if (params) {
    Object.keys(params).sort().forEach((key) => {
      if (key) {
        url.searchParams.set(key, params[key]);
      }
    });
  }
  return url.href;
}
const AGE_HEADER_LINE_REGEX = /^age:\s*[\d.]+\s*$/im;
class XhrLoader {
  constructor(config) {
    this.xhrSetup = void 0;
    this.requestTimeout = void 0;
    this.retryTimeout = void 0;
    this.retryDelay = void 0;
    this.config = null;
    this.callbacks = null;
    this.context = null;
    this.loader = null;
    this.stats = void 0;
    this.xhrSetup = config ? config.xhrSetup || null : null;
    this.stats = new LoadStats();
    this.retryDelay = 0;
  }
  destroy() {
    this.callbacks = null;
    this.abortInternal();
    this.loader = null;
    this.config = null;
    this.context = null;
    this.xhrSetup = null;
  }
  abortInternal() {
    const loader = this.loader;
    self.clearTimeout(this.requestTimeout);
    self.clearTimeout(this.retryTimeout);
    if (loader) {
      loader.onreadystatechange = null;
      loader.onprogress = null;
      if (loader.readyState !== 4) {
        this.stats.aborted = true;
        loader.abort();
      }
    }
  }
  abort() {
    var _this$callbacks;
    this.abortInternal();
    if ((_this$callbacks = this.callbacks) != null && _this$callbacks.onAbort) {
      this.callbacks.onAbort(this.stats, this.context, this.loader);
    }
  }
  load(context, config, callbacks) {
    if (this.stats.loading.start) {
      throw new Error("Loader can only be used once.");
    }
    this.stats.loading.start = self.performance.now();
    this.context = context;
    this.config = config;
    this.callbacks = callbacks;
    this.loadInternal();
  }
  loadInternal() {
    const {
      config,
      context
    } = this;
    if (!config || !context) {
      return;
    }
    const xhr = this.loader = new self.XMLHttpRequest();
    const stats = this.stats;
    stats.loading.first = 0;
    stats.loaded = 0;
    stats.aborted = false;
    const xhrSetup = this.xhrSetup;
    if (xhrSetup) {
      Promise.resolve().then(() => {
        if (this.loader !== xhr || this.stats.aborted) return;
        return xhrSetup(xhr, context.url);
      }).catch((error) => {
        if (this.loader !== xhr || this.stats.aborted) return;
        xhr.open("GET", context.url, true);
        return xhrSetup(xhr, context.url);
      }).then(() => {
        if (this.loader !== xhr || this.stats.aborted) return;
        this.openAndSendXhr(xhr, context, config);
      }).catch((error) => {
        this.callbacks.onError({
          code: xhr.status,
          text: error.message
        }, context, xhr, stats);
        return;
      });
    } else {
      this.openAndSendXhr(xhr, context, config);
    }
  }
  openAndSendXhr(xhr, context, config) {
    if (!xhr.readyState) {
      xhr.open("GET", context.url, true);
    }
    const headers = context.headers;
    const {
      maxTimeToFirstByteMs,
      maxLoadTimeMs
    } = config.loadPolicy;
    if (headers) {
      for (const header in headers) {
        xhr.setRequestHeader(header, headers[header]);
      }
    }
    if (context.rangeEnd) {
      xhr.setRequestHeader("Range", "bytes=" + context.rangeStart + "-" + (context.rangeEnd - 1));
    }
    xhr.onreadystatechange = this.readystatechange.bind(this);
    xhr.onprogress = this.loadprogress.bind(this);
    xhr.responseType = context.responseType;
    self.clearTimeout(this.requestTimeout);
    config.timeout = maxTimeToFirstByteMs && isFiniteNumber(maxTimeToFirstByteMs) ? maxTimeToFirstByteMs : maxLoadTimeMs;
    this.requestTimeout = self.setTimeout(this.loadtimeout.bind(this), config.timeout);
    xhr.send();
  }
  readystatechange() {
    const {
      context,
      loader: xhr,
      stats
    } = this;
    if (!context || !xhr) {
      return;
    }
    const readyState = xhr.readyState;
    const config = this.config;
    if (stats.aborted) {
      return;
    }
    if (readyState >= 2) {
      if (stats.loading.first === 0) {
        stats.loading.first = Math.max(self.performance.now(), stats.loading.start);
        if (config.timeout !== config.loadPolicy.maxLoadTimeMs) {
          self.clearTimeout(this.requestTimeout);
          config.timeout = config.loadPolicy.maxLoadTimeMs;
          this.requestTimeout = self.setTimeout(this.loadtimeout.bind(this), config.loadPolicy.maxLoadTimeMs - (stats.loading.first - stats.loading.start));
        }
      }
      if (readyState === 4) {
        self.clearTimeout(this.requestTimeout);
        xhr.onreadystatechange = null;
        xhr.onprogress = null;
        const status2 = xhr.status;
        const useResponse = xhr.responseType !== "text";
        if (status2 >= 200 && status2 < 300 && (useResponse && xhr.response || xhr.responseText !== null)) {
          stats.loading.end = Math.max(self.performance.now(), stats.loading.first);
          const data = useResponse ? xhr.response : xhr.responseText;
          const len = xhr.responseType === "arraybuffer" ? data.byteLength : data.length;
          stats.loaded = stats.total = len;
          stats.bwEstimate = stats.total * 8e3 / (stats.loading.end - stats.loading.first);
          if (!this.callbacks) {
            return;
          }
          const onProgress = this.callbacks.onProgress;
          if (onProgress) {
            onProgress(stats, context, data, xhr);
          }
          if (!this.callbacks) {
            return;
          }
          const response = {
            url: xhr.responseURL,
            data,
            code: status2
          };
          this.callbacks.onSuccess(response, stats, context, xhr);
        } else {
          const retryConfig = config.loadPolicy.errorRetry;
          const retryCount = stats.retry;
          const response = {
            url: context.url,
            data: void 0,
            code: status2
          };
          if (shouldRetry(retryConfig, retryCount, false, response)) {
            this.retry(retryConfig);
          } else {
            logger.error(`${status2} while loading ${context.url}`);
            this.callbacks.onError({
              code: status2,
              text: xhr.statusText
            }, context, xhr, stats);
          }
        }
      }
    }
  }
  loadtimeout() {
    if (!this.config) return;
    const retryConfig = this.config.loadPolicy.timeoutRetry;
    const retryCount = this.stats.retry;
    if (shouldRetry(retryConfig, retryCount, true)) {
      this.retry(retryConfig);
    } else {
      var _this$context;
      logger.warn(`timeout while loading ${(_this$context = this.context) == null ? void 0 : _this$context.url}`);
      const callbacks = this.callbacks;
      if (callbacks) {
        this.abortInternal();
        callbacks.onTimeout(this.stats, this.context, this.loader);
      }
    }
  }
  retry(retryConfig) {
    const {
      context,
      stats
    } = this;
    this.retryDelay = getRetryDelay(retryConfig, stats.retry);
    stats.retry++;
    logger.warn(`${status ? "HTTP Status " + status : "Timeout"} while loading ${context == null ? void 0 : context.url}, retrying ${stats.retry}/${retryConfig.maxNumRetry} in ${this.retryDelay}ms`);
    this.abortInternal();
    this.loader = null;
    self.clearTimeout(this.retryTimeout);
    this.retryTimeout = self.setTimeout(this.loadInternal.bind(this), this.retryDelay);
  }
  loadprogress(event) {
    const stats = this.stats;
    stats.loaded = event.loaded;
    if (event.lengthComputable) {
      stats.total = event.total;
    }
  }
  getCacheAge() {
    let result = null;
    if (this.loader && AGE_HEADER_LINE_REGEX.test(this.loader.getAllResponseHeaders())) {
      const ageHeader = this.loader.getResponseHeader("age");
      result = ageHeader ? parseFloat(ageHeader) : null;
    }
    return result;
  }
  getResponseHeader(name) {
    if (this.loader && new RegExp(`^${name}:\\s*[\\d.]+\\s*$`, "im").test(this.loader.getAllResponseHeaders())) {
      return this.loader.getResponseHeader(name);
    }
    return null;
  }
}
const WHITESPACE_CHAR = /\s/;
const Cues = {
  newCue(track, startTime, endTime, captionScreen) {
    const result = [];
    let row;
    let cue;
    let indenting;
    let indent;
    let text;
    const Cue = self.VTTCue || self.TextTrackCue;
    for (let r2 = 0; r2 < captionScreen.rows.length; r2++) {
      row = captionScreen.rows[r2];
      indenting = true;
      indent = 0;
      text = "";
      if (!row.isEmpty()) {
        var _track$cues;
        for (let c = 0; c < row.chars.length; c++) {
          if (WHITESPACE_CHAR.test(row.chars[c].uchar) && indenting) {
            indent++;
          } else {
            text += row.chars[c].uchar;
            indenting = false;
          }
        }
        row.cueStartTime = startTime;
        if (startTime === endTime) {
          endTime += 1e-4;
        }
        if (indent >= 16) {
          indent--;
        } else {
          indent++;
        }
        const cueText = fixLineBreaks(text.trim());
        const id = generateCueId(startTime, endTime, cueText);
        if (!(track != null && (_track$cues = track.cues) != null && _track$cues.getCueById(id))) {
          cue = new Cue(startTime, endTime, cueText);
          cue.id = id;
          cue.line = r2 + 1;
          cue.align = "left";
          cue.position = 10 + Math.min(80, Math.floor(indent * 8 / 32) * 10);
          result.push(cue);
        }
      }
    }
    if (track && result.length) {
      result.sort((cueA, cueB) => {
        if (cueA.line === "auto" || cueB.line === "auto") {
          return 0;
        }
        if (cueA.line > 8 && cueB.line > 8) {
          return cueB.line - cueA.line;
        }
        return cueA.line - cueB.line;
      });
      result.forEach((cue2) => addCueToTrack(track, cue2));
    }
    return result;
  }
};
const defaultLoadPolicy = {
  maxTimeToFirstByteMs: 8e3,
  maxLoadTimeMs: 2e4,
  timeoutRetry: null,
  errorRetry: null
};
_objectSpread2(_objectSpread2({
  autoStartLoad: true,
  // used by stream-controller
  startPosition: -1,
  // used by stream-controller
  defaultAudioCodec: void 0,
  // used by stream-controller
  debug: false,
  // used by logger
  capLevelOnFPSDrop: false,
  // used by fps-controller
  capLevelToPlayerSize: false,
  // used by cap-level-controller
  ignoreDevicePixelRatio: false,
  // used by cap-level-controller
  preferManagedMediaSource: true,
  initialLiveManifestSize: 1,
  // used by stream-controller
  maxBufferLength: 30,
  // used by stream-controller
  backBufferLength: Infinity,
  // used by buffer-controller
  frontBufferFlushThreshold: Infinity,
  maxBufferSize: 60 * 1e3 * 1e3,
  // used by stream-controller
  maxBufferHole: 0.1,
  // used by stream-controller
  highBufferWatchdogPeriod: 2,
  // used by stream-controller
  nudgeOffset: 0.1,
  // used by stream-controller
  nudgeMaxRetry: 3,
  // used by stream-controller
  maxFragLookUpTolerance: 0.25,
  // used by stream-controller
  liveSyncDurationCount: 3,
  // used by latency-controller
  liveMaxLatencyDurationCount: Infinity,
  // used by latency-controller
  liveSyncDuration: void 0,
  // used by latency-controller
  liveMaxLatencyDuration: void 0,
  // used by latency-controller
  maxLiveSyncPlaybackRate: 1,
  // used by latency-controller
  liveDurationInfinity: false,
  // used by buffer-controller
  /**
   * @deprecated use backBufferLength
   */
  liveBackBufferLength: null,
  // used by buffer-controller
  maxMaxBufferLength: 600,
  // used by stream-controller
  enableWorker: true,
  // used by transmuxer
  workerPath: null,
  // used by transmuxer
  enableSoftwareAES: true,
  // used by decrypter
  startLevel: void 0,
  // used by level-controller
  startFragPrefetch: false,
  // used by stream-controller
  fpsDroppedMonitoringPeriod: 5e3,
  // used by fps-controller
  fpsDroppedMonitoringThreshold: 0.2,
  // used by fps-controller
  appendErrorMaxRetry: 3,
  // used by buffer-controller
  loader: XhrLoader,
  // loader: FetchLoader,
  fLoader: void 0,
  // used by fragment-loader
  pLoader: void 0,
  // used by playlist-loader
  xhrSetup: void 0,
  // used by xhr-loader
  licenseXhrSetup: void 0,
  // used by eme-controller
  licenseResponseCallback: void 0,
  // used by eme-controller
  abrController: AbrController,
  bufferController: BufferController,
  capLevelController: CapLevelController,
  errorController: ErrorController,
  fpsController: FPSController,
  stretchShortVideoTrack: false,
  // used by mp4-remuxer
  maxAudioFramesDrift: 1,
  // used by mp4-remuxer
  forceKeyFrameOnDiscontinuity: true,
  // used by ts-demuxer
  abrEwmaFastLive: 3,
  // used by abr-controller
  abrEwmaSlowLive: 9,
  // used by abr-controller
  abrEwmaFastVoD: 3,
  // used by abr-controller
  abrEwmaSlowVoD: 9,
  // used by abr-controller
  abrEwmaDefaultEstimate: 5e5,
  // 500 kbps  // used by abr-controller
  abrEwmaDefaultEstimateMax: 5e6,
  // 5 mbps
  abrBandWidthFactor: 0.95,
  // used by abr-controller
  abrBandWidthUpFactor: 0.7,
  // used by abr-controller
  abrMaxWithRealBitrate: false,
  // used by abr-controller
  maxStarvationDelay: 4,
  // used by abr-controller
  maxLoadingDelay: 4,
  // used by abr-controller
  minAutoBitrate: 0,
  // used by hls
  emeEnabled: false,
  // used by eme-controller
  widevineLicenseUrl: void 0,
  // used by eme-controller
  drmSystems: {},
  // used by eme-controller
  drmSystemOptions: {},
  // used by eme-controller
  requestMediaKeySystemAccessFunc: requestMediaKeySystemAccess,
  // used by eme-controller
  testBandwidth: true,
  progressive: false,
  lowLatencyMode: true,
  cmcd: void 0,
  enableDateRangeMetadataCues: true,
  enableEmsgMetadataCues: true,
  enableID3MetadataCues: true,
  useMediaCapabilities: true,
  certLoadPolicy: {
    default: defaultLoadPolicy
  },
  keyLoadPolicy: {
    default: {
      maxTimeToFirstByteMs: 8e3,
      maxLoadTimeMs: 2e4,
      timeoutRetry: {
        maxNumRetry: 1,
        retryDelayMs: 1e3,
        maxRetryDelayMs: 2e4,
        backoff: "linear"
      },
      errorRetry: {
        maxNumRetry: 8,
        retryDelayMs: 1e3,
        maxRetryDelayMs: 2e4,
        backoff: "linear"
      }
    }
  },
  manifestLoadPolicy: {
    default: {
      maxTimeToFirstByteMs: Infinity,
      maxLoadTimeMs: 2e4,
      timeoutRetry: {
        maxNumRetry: 2,
        retryDelayMs: 0,
        maxRetryDelayMs: 0
      },
      errorRetry: {
        maxNumRetry: 1,
        retryDelayMs: 1e3,
        maxRetryDelayMs: 8e3
      }
    }
  },
  playlistLoadPolicy: {
    default: {
      maxTimeToFirstByteMs: 1e4,
      maxLoadTimeMs: 2e4,
      timeoutRetry: {
        maxNumRetry: 2,
        retryDelayMs: 0,
        maxRetryDelayMs: 0
      },
      errorRetry: {
        maxNumRetry: 2,
        retryDelayMs: 1e3,
        maxRetryDelayMs: 8e3
      }
    }
  },
  fragLoadPolicy: {
    default: {
      maxTimeToFirstByteMs: 1e4,
      maxLoadTimeMs: 12e4,
      timeoutRetry: {
        maxNumRetry: 4,
        retryDelayMs: 0,
        maxRetryDelayMs: 0
      },
      errorRetry: {
        maxNumRetry: 6,
        retryDelayMs: 1e3,
        maxRetryDelayMs: 8e3
      }
    }
  },
  steeringManifestLoadPolicy: {
    default: {
      maxTimeToFirstByteMs: 1e4,
      maxLoadTimeMs: 2e4,
      timeoutRetry: {
        maxNumRetry: 2,
        retryDelayMs: 0,
        maxRetryDelayMs: 0
      },
      errorRetry: {
        maxNumRetry: 1,
        retryDelayMs: 1e3,
        maxRetryDelayMs: 8e3
      }
    }
  },
  // These default settings are deprecated in favor of the above policies
  // and are maintained for backwards compatibility
  manifestLoadingTimeOut: 1e4,
  manifestLoadingMaxRetry: 1,
  manifestLoadingRetryDelay: 1e3,
  manifestLoadingMaxRetryTimeout: 64e3,
  levelLoadingTimeOut: 1e4,
  levelLoadingMaxRetry: 4,
  levelLoadingRetryDelay: 1e3,
  levelLoadingMaxRetryTimeout: 64e3,
  fragLoadingTimeOut: 2e4,
  fragLoadingMaxRetry: 6,
  fragLoadingRetryDelay: 1e3,
  fragLoadingMaxRetryTimeout: 64e3
}, timelineConfig()), {}, {
  subtitleStreamController: SubtitleStreamController,
  subtitleTrackController: SubtitleTrackController,
  timelineController: TimelineController,
  audioStreamController: AudioStreamController,
  audioTrackController: AudioTrackController,
  emeController: EMEController,
  cmcdController: CMCDController,
  contentSteeringController: ContentSteeringController
});
function timelineConfig() {
  return {
    cueHandler: Cues,
    // used by timeline-controller
    enableWebVTT: true,
    // used by timeline-controller
    enableIMSC1: true,
    // used by timeline-controller
    enableCEA708Captions: true,
    // used by timeline-controller
    captionsTextTrack1Label: "English",
    // used by timeline-controller
    captionsTextTrack1LanguageCode: "en",
    // used by timeline-controller
    captionsTextTrack2Label: "Spanish",
    // used by timeline-controller
    captionsTextTrack2LanguageCode: "es",
    // used by timeline-controller
    captionsTextTrack3Label: "Unknown CC",
    // used by timeline-controller
    captionsTextTrack3LanguageCode: "",
    // used by timeline-controller
    captionsTextTrack4Label: "Unknown CC",
    // used by timeline-controller
    captionsTextTrack4LanguageCode: "",
    // used by timeline-controller
    renderTextTracksNatively: true
  };
}
function AudioPlayer($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      value = null,
      subtitles = null,
      label,
      i18n,
      dispatch_blob = () => Promise.resolve(),
      interactive = false,
      editable = true,
      trim_region_settings = {},
      waveform_settings,
      waveform_options,
      mode = void 0,
      loop,
      handle_reset_value = () => {
      },
      playback_position = void 0,
      onstop,
      onplay,
      onpause,
      onedit,
      onload
    } = $$props;
    value?.url;
    let container;
    let waveform;
    let playing = false;
    let audio_duration = 0;
    let trimDuration = 0;
    let show_volume_slider = false;
    let subtitles_toggle = true;
    let use_waveform = waveform_options.show_recording_waveform && !value?.is_stream;
    const handle_trim_audio = async (start, end) => {
      mode = "";
      onedit?.();
    };
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      $$renderer3.push(`<audio${attr_class("standard-player svelte-1ffmt2w", void 0, { "hidden": use_waveform })} controls${attr("autoplay", waveform_settings.autoplay, true)} preload="metadata"></audio> `);
      if (value === null) {
        $$renderer3.push("<!--[-->");
        Empty($$renderer3, {
          size: "small",
          children: ($$renderer4) => {
            Music($$renderer4);
          },
          $$slots: { default: true }
        });
      } else {
        $$renderer3.push("<!--[!-->");
        if (use_waveform) {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<div class="component-wrapper svelte-1ffmt2w"${attr("data-testid", label ? "waveform-" + label : "unlabelled-audio")}><div class="waveform-container svelte-1ffmt2w"><div id="waveform" class="svelte-1ffmt2w"${attr_style("", { height: "58px" })}></div></div> <div class="timestamps svelte-1ffmt2w"><time id="time" class="svelte-1ffmt2w">0:00</time> <div>`);
          if (mode === "edit" && trimDuration > 0) {
            $$renderer3.push("<!--[-->");
            $$renderer3.push(`<time id="trim-duration" class="svelte-1ffmt2w">${escape_html(format_time(trimDuration))}</time>`);
          } else {
            $$renderer3.push("<!--[!-->");
          }
          $$renderer3.push(`<!--]--> <time id="duration" class="svelte-1ffmt2w">0:00</time></div></div> <div class="subtitle-display svelte-1ffmt2w" data-testid="subtitle-display"></div> `);
          WaveformControls($$renderer3, {
            container,
            waveform,
            playing,
            audio_duration,
            i18n,
            interactive,
            handle_trim_audio,
            show_redo: interactive,
            handle_reset_value,
            waveform_options,
            trim_region_settings,
            editable,
            show_subtitles: subtitles !== null,
            get mode() {
              return mode;
            },
            set mode($$value) {
              mode = $$value;
              $$settled = false;
            },
            get trimDuration() {
              return trimDuration;
            },
            set trimDuration($$value) {
              trimDuration = $$value;
              $$settled = false;
            },
            get show_volume_slider() {
              return show_volume_slider;
            },
            set show_volume_slider($$value) {
              show_volume_slider = $$value;
              $$settled = false;
            },
            get subtitles_toggle() {
              return subtitles_toggle;
            },
            set subtitles_toggle($$value) {
              subtitles_toggle = $$value;
              $$settled = false;
            }
          });
          $$renderer3.push(`<!----></div>`);
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]-->`);
      }
      $$renderer3.push(`<!--]-->`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { mode, playback_position });
  });
}
function StaticAudio($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      value = null,
      subtitles = null,
      label,
      show_label = true,
      buttons = [],
      on_custom_button_click = null,
      i18n,
      waveform_settings = {},
      waveform_options = { show_recording_waveform: true },
      editable = true,
      loop,
      display_icon_button_wrapper_top_corner = false,
      minimal = false,
      playback_position = void 0,
      onchange,
      onplay,
      onpause,
      onstop,
      onshare,
      onerror
    } = $$props;
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      BlockLabel($$renderer3, {
        show_label,
        Icon: Music,
        float: false,
        label: label || i18n("audio.audio")
      });
      $$renderer3.push(`<!----> `);
      if (value !== null) {
        $$renderer3.push("<!--[-->");
        if (minimal) {
          $$renderer3.push("<!--[-->");
          MinimalAudioPlayer($$renderer3, { value, label: label || i18n("audio.audio"), loop });
        } else {
          $$renderer3.push("<!--[!-->");
          IconButtonWrapper($$renderer3, {
            display_top_corner: display_icon_button_wrapper_top_corner,
            buttons,
            on_custom_button_click,
            children: ($$renderer4) => {
              if (buttons.some((btn) => typeof btn === "string" && btn === "download")) {
                $$renderer4.push("<!--[-->");
                DownloadLink($$renderer4, {
                  href: value.is_stream ? value.url?.replace("playlist.m3u8", "playlist-file") : value.url,
                  download: value.orig_name || value.path,
                  children: ($$renderer5) => {
                    IconButton($$renderer5, { Icon: Download, label: i18n("common.download") });
                  },
                  $$slots: { default: true }
                });
              } else {
                $$renderer4.push("<!--[!-->");
              }
              $$renderer4.push(`<!--]--> `);
              if (buttons.some((btn) => typeof btn === "string" && btn === "share")) {
                $$renderer4.push("<!--[-->");
                ShareButton($$renderer4, {
                  i18n,
                  onerror,
                  onshare,
                  formatter: async (fileData) => {
                    if (!fileData || !fileData.url) return "";
                    let url = await uploadToHuggingFace(fileData.url);
                    return `<audio controls src="${url}"></audio>`;
                  },
                  value
                });
              } else {
                $$renderer4.push("<!--[!-->");
              }
              $$renderer4.push(`<!--]-->`);
            }
          });
          $$renderer3.push(`<!----> `);
          AudioPlayer($$renderer3, {
            value,
            subtitles: Array.isArray(subtitles) ? subtitles : subtitles?.url,
            label,
            i18n,
            waveform_settings,
            waveform_options,
            editable,
            loop,
            onpause,
            onplay,
            onstop,
            onload: () => {
            },
            get playback_position() {
              return playback_position;
            },
            set playback_position($$value) {
              playback_position = $$value;
              $$settled = false;
            }
          });
          $$renderer3.push(`<!---->`);
        }
        $$renderer3.push(`<!--]-->`);
      } else {
        $$renderer3.push("<!--[!-->");
        Empty($$renderer3, {
          size: "small",
          children: ($$renderer4) => {
            Music($$renderer4);
          },
          $$slots: { default: true }
        });
      }
      $$renderer3.push(`<!--]-->`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { playback_position });
  });
}
const StaticAudio$1 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: StaticAudio
}, Symbol.toStringTag, { value: "Module" }));

export { AudioPlayer as A, StaticAudio as S, StaticAudio$1 as a };
//# sourceMappingURL=StaticAudio-CGENs1q8.js.map
