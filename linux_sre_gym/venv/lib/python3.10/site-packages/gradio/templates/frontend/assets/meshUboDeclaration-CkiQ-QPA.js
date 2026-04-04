import { S as ShaderStore } from './index-CxR0Yf_7.js';

// Do not edit.
const name = "meshUboDeclaration";
const shader = `struct Mesh {world : mat4x4<f32>,
visibility : f32,};var<uniform> mesh : Mesh;
#define WORLD_UBO
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStoreWGSL[name]) {
    ShaderStore.IncludesShadersStoreWGSL[name] = shader;
}
//# sourceMappingURL=meshUboDeclaration-CkiQ-QPA.js.map
