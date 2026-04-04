import './async-D55cHugf.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { B as Button } from './Button-C8geAOuh.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './index-u8mz_F03.js';
import './context-CBkBucIx.js';
import './Image-BLyvTcTy.js';

function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { $$slots, $$events, ..._props } = $$props;
    const gradio = new Gradio(_props);
    function handle_click() {
      gradio.dispatch("click");
    }
    Button($$renderer2, {
      value: gradio.props.value,
      variant: gradio.props.variant,
      elem_id: gradio.shared.elem_id,
      elem_classes: gradio.shared.elem_classes,
      size: gradio.props.size,
      scale: gradio.shared.scale,
      link: gradio.props.link,
      icon: gradio.props.icon,
      min_width: gradio.shared.min_width,
      visible: gradio.shared.visible,
      disabled: !gradio.shared.interactive,
      link_target: gradio.props.link_target,
      onclick: handle_click,
      children: ($$renderer3) => {
        $$renderer3.push(`<!---->${escape_html(gradio.props.value ?? "")}`);
      }
    });
  });
}

export { Button as BaseButton, Index as default };
//# sourceMappingURL=Index7-D46XnVSh.js.map
