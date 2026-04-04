import { ab as listen, ac as without_reactive_context } from './index-BvBk1Iap.js';

/**
 * @param {'innerWidth' | 'innerHeight' | 'outerWidth' | 'outerHeight'} type
 * @param {(size: number) => void} set
 */
function bind_window_size(type, set) {
	listen(window, ['resize'], () => without_reactive_context(() => set(window[type])));
}

export { bind_window_size as b };
//# sourceMappingURL=window-BGXjFJtG.js.map
