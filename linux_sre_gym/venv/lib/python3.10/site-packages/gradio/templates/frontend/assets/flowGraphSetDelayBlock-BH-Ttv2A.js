import { a as FlowGraphAsyncExecutionBlock } from './KHR_interactivity-BPhVhH5L.js';
import { b as RichTypeNumber, i as RichTypeFlowGraphInteger, F as FlowGraphInteger } from './declarationMapper-eK_q7dJF.js';
import { O as Observable, L as Logger, R as RegisterClass } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';
import './objectModelMapping-DnUo8NMz.js';

/**
 * The current state of the timer
 */
var TimerState;
(function (TimerState) {
    /**
     * Timer initialized, not yet started
     */
    TimerState[TimerState["INIT"] = 0] = "INIT";
    /**
     * Timer started and counting
     */
    TimerState[TimerState["STARTED"] = 1] = "STARTED";
    /**
     * Timer ended (whether aborted or time reached)
     */
    TimerState[TimerState["ENDED"] = 2] = "ENDED";
})(TimerState || (TimerState = {}));
/**
 * An advanced implementation of a timer class
 */
class AdvancedTimer {
    /**
     * Will construct a new advanced timer based on the options provided. Timer will not start until start() is called.
     * @param options construction options for this advanced timer
     */
    constructor(options) {
        /**
         * Will notify each time the timer calculates the remaining time
         */
        this.onEachCountObservable = new Observable();
        /**
         * Will trigger when the timer was aborted due to the break condition
         */
        this.onTimerAbortedObservable = new Observable();
        /**
         * Will trigger when the timer ended successfully
         */
        this.onTimerEndedObservable = new Observable();
        /**
         * Will trigger when the timer state has changed
         */
        this.onStateChangedObservable = new Observable();
        this._observer = null;
        this._breakOnNextTick = false;
        this._tick = (payload) => {
            const now = Date.now();
            this._timer = now - this._startTime;
            const data = {
                startTime: this._startTime,
                currentTime: now,
                deltaTime: this._timer,
                completeRate: this._timer / this._timeToEnd,
                payload,
            };
            const shouldBreak = this._breakOnNextTick || this._breakCondition(data);
            if (shouldBreak || this._timer >= this._timeToEnd) {
                this._stop(data, shouldBreak);
            }
            else {
                this.onEachCountObservable.notifyObservers(data);
            }
        };
        this._setState(0 /* TimerState.INIT */);
        this._contextObservable = options.contextObservable;
        this._observableParameters = options.observableParameters ?? {};
        this._breakCondition = options.breakCondition ?? (() => false);
        this._timeToEnd = options.timeout;
        if (options.onEnded) {
            this.onTimerEndedObservable.add(options.onEnded);
        }
        if (options.onTick) {
            this.onEachCountObservable.add(options.onTick);
        }
        if (options.onAborted) {
            this.onTimerAbortedObservable.add(options.onAborted);
        }
    }
    /**
     * set a breaking condition for this timer. Default is to never break during count
     * @param predicate the new break condition. Returns true to break, false otherwise
     */
    set breakCondition(predicate) {
        this._breakCondition = predicate;
    }
    /**
     * Reset ALL associated observables in this advanced timer
     */
    clearObservables() {
        this.onEachCountObservable.clear();
        this.onTimerAbortedObservable.clear();
        this.onTimerEndedObservable.clear();
        this.onStateChangedObservable.clear();
    }
    /**
     * Will start a new iteration of this timer. Only one instance of this timer can run at a time.
     *
     * @param timeToEnd how much time to measure until timer ended
     */
    start(timeToEnd = this._timeToEnd) {
        if (this._state === 1 /* TimerState.STARTED */) {
            throw new Error("Timer already started. Please stop it before starting again");
        }
        this._timeToEnd = timeToEnd;
        this._startTime = Date.now();
        this._timer = 0;
        this._observer = this._contextObservable.add(this._tick, this._observableParameters.mask, this._observableParameters.insertFirst, this._observableParameters.scope);
        this._setState(1 /* TimerState.STARTED */);
    }
    /**
     * Will force a stop on the next tick.
     */
    stop() {
        if (this._state !== 1 /* TimerState.STARTED */) {
            return;
        }
        this._breakOnNextTick = true;
    }
    /**
     * Dispose this timer, clearing all resources
     */
    dispose() {
        if (this._observer) {
            this._contextObservable.remove(this._observer);
        }
        this.clearObservables();
    }
    _setState(newState) {
        this._state = newState;
        this.onStateChangedObservable.notifyObservers(this._state);
    }
    _stop(data, aborted = false) {
        this._contextObservable.remove(this._observer);
        this._setState(2 /* TimerState.ENDED */);
        if (aborted) {
            this.onTimerAbortedObservable.notifyObservers(data);
        }
        else {
            this.onTimerEndedObservable.notifyObservers(data);
        }
    }
}

/**
 * Block that sets a delay in seconds before activating the output signal.
 */
class FlowGraphSetDelayBlock extends FlowGraphAsyncExecutionBlock {
    constructor(config) {
        super(config);
        this.cancel = this._registerSignalInput("cancel");
        this.duration = this.registerDataInput("duration", RichTypeNumber);
        this.lastDelayIndex = this.registerDataOutput("lastDelayIndex", RichTypeFlowGraphInteger, new FlowGraphInteger(-1));
    }
    _preparePendingTasks(context) {
        const duration = this.duration.getValue(context);
        if (duration < 0 || isNaN(duration) || !isFinite(duration)) {
            return this._reportError(context, "Invalid duration in SetDelay block");
        }
        // active delays are global to the context
        const activeDelays = context._getGlobalContextVariable("activeDelays", 0);
        if (activeDelays >= FlowGraphSetDelayBlock.MaxParallelDelayCount) {
            return this._reportError(context, "Max parallel delays reached");
        }
        // get the last global delay index
        const lastDelayIndex = context._getGlobalContextVariable("lastDelayIndex", -1);
        // these are block-specific and not global
        const timers = context._getExecutionVariable(this, "pendingDelays", []);
        const scene = context.configuration.scene;
        const timer = new AdvancedTimer({
            timeout: duration * 1000, // duration is in seconds
            contextObservable: scene.onBeforeRenderObservable,
            onEnded: () => this._onEnded(timer, context),
        });
        timer.start();
        const newIndex = lastDelayIndex + 1;
        this.lastDelayIndex.setValue(new FlowGraphInteger(newIndex), context);
        context._setGlobalContextVariable("lastDelayIndex", newIndex);
        timers[newIndex] = timer;
        context._setExecutionVariable(this, "pendingDelays", timers);
        this._updateGlobalTimers(context);
    }
    _cancelPendingTasks(context) {
        const timers = context._getExecutionVariable(this, "pendingDelays", []);
        for (const timer of timers) {
            timer?.dispose();
        }
        context._deleteExecutionVariable(this, "pendingDelays");
        this.lastDelayIndex.setValue(new FlowGraphInteger(-1), context);
        this._updateGlobalTimers(context);
    }
    _execute(context, callingSignal) {
        if (callingSignal === this.cancel) {
            this._cancelPendingTasks(context);
            return;
        }
        else {
            this._preparePendingTasks(context);
            this.out._activateSignal(context);
        }
    }
    getClassName() {
        return "FlowGraphSetDelayBlock" /* FlowGraphBlockNames.SetDelay */;
    }
    _onEnded(timer, context) {
        const timers = context._getExecutionVariable(this, "pendingDelays", []);
        const index = timers.indexOf(timer);
        if (index !== -1) {
            timers.splice(index, 1);
        }
        else {
            Logger.Warn("FlowGraphTimerBlock: Timer ended but was not found in the running timers list");
        }
        context._removePendingBlock(this);
        this.done._activateSignal(context);
        this._updateGlobalTimers(context);
    }
    _updateGlobalTimers(context) {
        const timers = context._getExecutionVariable(this, "pendingDelays", []);
        const globalTimers = context._getGlobalContextVariable("pendingDelays", []);
        // there should NEVER be the same index in the global and local timers, unless they are equal
        for (let i = 0; i < timers.length; i++) {
            if (!timers[i]) {
                continue;
            }
            const timer = timers[i];
            if (globalTimers[i] && globalTimers[i] !== timer) {
                Logger.Warn("FlowGraphTimerBlock: Timer ended but was not found in the running timers list");
            }
            else {
                globalTimers[i] = timer;
            }
        }
        context._setGlobalContextVariable("pendingDelays", globalTimers);
    }
}
/**
 * The maximum number of parallel delays that can be set per node.
 */
FlowGraphSetDelayBlock.MaxParallelDelayCount = 100;
RegisterClass("FlowGraphSetDelayBlock" /* FlowGraphBlockNames.SetDelay */, FlowGraphSetDelayBlock);

export { FlowGraphSetDelayBlock };
//# sourceMappingURL=flowGraphSetDelayBlock-BH-Ttv2A.js.map
