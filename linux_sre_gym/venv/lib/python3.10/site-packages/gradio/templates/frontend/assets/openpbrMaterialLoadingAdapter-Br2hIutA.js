/**
 * Material Loading Adapter for OpenPBR materials that provides a unified OpenPBR-like interface.
 */
class OpenPBRMaterialLoadingAdapter {
    /**
     * Creates a new instance of the OpenPBRMaterialLoadingAdapter.
     * @param material - The OpenPBR material to adapt.
     */
    constructor(material) {
        this._material = material;
    }
    /**
     * Gets the underlying material
     */
    get material() {
        return this._material;
    }
    /**
     * Whether the material should be treated as unlit
     */
    get isUnlit() {
        return this._material.unlit;
    }
    /**
     * Sets whether the material should be treated as unlit
     */
    set isUnlit(value) {
        this._material.unlit = value;
    }
    // ========================================
    // CULLING PROPERTIES
    // ========================================
    /**
     * Sets whether back face culling is enabled.
     * @param value True to enable back face culling
     */
    set backFaceCulling(value) {
        this._material.backFaceCulling = value;
    }
    /**
     * Gets whether back face culling is enabled.
     * @returns True if back face culling is enabled
     */
    get backFaceCulling() {
        return this._material.backFaceCulling;
    }
    /**
     * Sets whether two-sided lighting is enabled.
     * @param value True to enable two-sided lighting
     */
    set twoSidedLighting(value) {
        this._material.twoSidedLighting = value;
    }
    /**
     * Gets whether two-sided lighting is enabled.
     * @returns True if two-sided lighting is enabled
     */
    get twoSidedLighting() {
        return this._material.twoSidedLighting;
    }
    // ========================================
    // ALPHA PROPERTIES
    // ========================================
    /**
     * Sets the alpha cutoff value for alpha testing.
     * Note: OpenPBR doesn't have a direct equivalent, so this is a no-op.
     * @param value The alpha cutoff threshold (ignored for OpenPBR)
     */
    set alphaCutOff(value) {
        // OpenPBR doesn't have a direct equivalent, but could be implemented if needed
    }
    /**
     * Gets the alpha cutoff value.
     * @returns Default value of 0.5 (OpenPBR doesn't support this directly)
     */
    get alphaCutOff() {
        return 0.5; // Default value
    }
    /**
     * Sets whether to use alpha from the base color texture.
     * Note: OpenPBR handles this differently through the baseColorTexture alpha channel.
     * @param value True to use alpha from base color texture (handled automatically in OpenPBR)
     */
    set useAlphaFromBaseColorTexture(value) {
        this._material._useAlphaFromBaseColorTexture = value;
    }
    /**
     * Gets whether alpha is used from the base color texture.
     * @returns Always false for OpenPBR as it's handled automatically
     */
    get useAlphaFromBaseColorTexture() {
        return false;
    }
    /**
     * Gets whether the transparency is treated as alpha coverage.
     */
    get transparencyAsAlphaCoverage() {
        // OpenPBR doesn't support treating transparency as alpha coverage.
        return false;
    }
    /**
     * Sets/Gets whether the transparency is treated as alpha coverage
     */
    set transparencyAsAlphaCoverage(value) {
        // OpenPBR doesn't support treating transparency as alpha coverage.
    }
    // ========================================
    // BASE PARAMETERS
    // ========================================
    /**
     * Sets the base color of the OpenPBR material.
     * @param value The base color as a Color3
     */
    set baseColor(value) {
        this._material.baseColor = value;
    }
    /**
     * Gets the base color of the OpenPBR material.
     * @returns The base color as a Color3
     */
    get baseColor() {
        return this._material.baseColor;
    }
    /**
     * Sets the base color texture of the OpenPBR material.
     * @param value The base color texture or null
     */
    set baseColorTexture(value) {
        this._material.baseColorTexture = value;
    }
    /**
     * Gets the base color texture of the OpenPBR material.
     * @returns The base color texture or null
     */
    get baseColorTexture() {
        return this._material.baseColorTexture;
    }
    /**
     * Sets the base diffuse roughness of the OpenPBR material.
     * @param value The diffuse roughness value (0-1)
     */
    set baseDiffuseRoughness(value) {
        this._material.baseDiffuseRoughness = value;
    }
    /**
     * Gets the base diffuse roughness of the OpenPBR material.
     * @returns The diffuse roughness value (0-1)
     */
    get baseDiffuseRoughness() {
        return this._material.baseDiffuseRoughness;
    }
    /**
     * Sets the base diffuse roughness texture of the OpenPBR material.
     * @param value The diffuse roughness texture or null
     */
    set baseDiffuseRoughnessTexture(value) {
        this._material.baseDiffuseRoughnessTexture = value;
    }
    /**
     * Gets the base diffuse roughness texture of the OpenPBR material.
     * @returns The diffuse roughness texture or null
     */
    get baseDiffuseRoughnessTexture() {
        return this._material.baseDiffuseRoughnessTexture;
    }
    /**
     * Sets the base metalness value of the OpenPBR material.
     * @param value The metalness value (0-1)
     */
    set baseMetalness(value) {
        this._material.baseMetalness = value;
    }
    /**
     * Gets the base metalness value of the OpenPBR material.
     * @returns The metalness value (0-1)
     */
    get baseMetalness() {
        return this._material.baseMetalness;
    }
    /**
     * Sets the base metalness texture of the OpenPBR material.
     * @param value The metalness texture or null
     */
    set baseMetalnessTexture(value) {
        this._material.baseMetalnessTexture = value;
    }
    /**
     * Gets the base metalness texture of the OpenPBR material.
     * @returns The metalness texture or null
     */
    get baseMetalnessTexture() {
        return this._material.baseMetalnessTexture;
    }
    /**
     * Sets whether to use roughness from the metallic texture's green channel.
     * @param value True to use green channel for roughness
     */
    set useRoughnessFromMetallicTextureGreen(value) {
        this._material._useRoughnessFromMetallicTextureGreen = value;
    }
    /**
     * Sets whether to use metalness from the metallic texture's blue channel.
     * @param value True to use blue channel for metalness
     */
    set useMetallicFromMetallicTextureBlue(value) {
        this._material._useMetallicFromMetallicTextureBlue = value;
    }
    // ========================================
    // SPECULAR PARAMETERS
    // ========================================
    /**
     * Configures specular properties for OpenPBR material.
     * @param _enableEdgeColor Whether to enable edge color support (ignored for OpenPBR)
     */
    enableSpecularEdgeColor(_enableEdgeColor = false) {
        // OpenPBR already supports edge color natively, no configuration needed
    }
    /**
     * Sets the specular weight of the OpenPBR material.
     * @param value The specular weight value (0-1)
     */
    set specularWeight(value) {
        this._material.specularWeight = value;
    }
    /**
     * Gets the specular weight of the OpenPBR material.
     * @returns The specular weight value (0-1)
     */
    get specularWeight() {
        return this._material.specularWeight;
    }
    /**
     * Sets the specular weight texture of the OpenPBR material.
     * If the same texture is used for specular color, optimizes by using alpha channel for weight.
     * @param value The specular weight texture or null
     */
    set specularWeightTexture(value) {
        if (this._material.specularColorTexture === value) {
            this._material.specularWeightTexture = null;
            this._material._useSpecularWeightFromSpecularColorTexture = true;
            this._material._useSpecularWeightFromAlpha = true;
        }
        else {
            this._material.specularWeightTexture = value;
        }
    }
    /**
     * Gets the specular weight texture of the OpenPBR material.
     * @returns The specular weight texture or null
     */
    get specularWeightTexture() {
        return this._material.specularWeightTexture;
    }
    /**
     * Sets the specular color of the OpenPBR material.
     * @param value The specular color as a Color3
     */
    set specularColor(value) {
        this._material.specularColor = value;
    }
    /**
     * Gets the specular color of the OpenPBR material.
     * @returns The specular color as a Color3
     */
    get specularColor() {
        return this._material.specularColor;
    }
    /**
     * Sets the specular color texture of the OpenPBR material.
     * If the same texture is used for specular weight, optimizes by using alpha channel for weight.
     * @param value The specular color texture or null
     */
    set specularColorTexture(value) {
        this._material.specularColorTexture = value;
        if (this._material.specularWeightTexture === this._material.specularColorTexture) {
            this._material.specularWeightTexture = null;
            this._material._useSpecularWeightFromSpecularColorTexture = true;
            this._material._useSpecularWeightFromAlpha = true;
        }
    }
    /**
     * Gets the specular color texture of the OpenPBR material.
     * @returns The specular color texture or null
     */
    get specularColorTexture() {
        return this._material.specularColorTexture;
    }
    /**
     * Sets the specular roughness of the OpenPBR material.
     * @param value The roughness value (0-1)
     */
    set specularRoughness(value) {
        this._material.specularRoughness = value;
    }
    /**
     * Gets the specular roughness of the OpenPBR material.
     * @returns The roughness value (0-1)
     */
    get specularRoughness() {
        return this._material.specularRoughness;
    }
    /**
     * Sets the specular roughness texture of the OpenPBR material.
     * @param value The roughness texture or null
     */
    set specularRoughnessTexture(value) {
        this._material.specularRoughnessTexture = value;
    }
    /**
     * Gets the specular roughness texture of the OpenPBR material.
     * @returns The roughness texture or null
     */
    get specularRoughnessTexture() {
        return this._material.specularRoughnessTexture;
    }
    /**
     * Sets the specular index of refraction (IOR) of the OpenPBR material.
     * @param value The IOR value
     */
    set specularIor(value) {
        this._material.specularIor = value;
    }
    /**
     * Gets the specular index of refraction (IOR) of the OpenPBR material.
     * @returns The IOR value
     */
    get specularIor() {
        return this._material.specularIor;
    }
    // ========================================
    // EMISSION PARAMETERS
    // ========================================
    /**
     * Sets the emission color of the OpenPBR material.
     * @param value The emission color as a Color3
     */
    set emissionColor(value) {
        this._material.emissionColor = value;
    }
    /**
     * Gets the emission color of the OpenPBR material.
     * @returns The emission color as a Color3
     */
    get emissionColor() {
        return this._material.emissionColor;
    }
    /**
     * Sets the emission luminance of the OpenPBR material.
     * @param value The emission luminance value
     */
    set emissionLuminance(value) {
        this._material.emissionLuminance = value;
    }
    /**
     * Gets the emission luminance of the OpenPBR material.
     * @returns The emission luminance value
     */
    get emissionLuminance() {
        return this._material.emissionLuminance;
    }
    /**
     * Sets the emission color texture of the OpenPBR material.
     * @param value The emission texture or null
     */
    set emissionColorTexture(value) {
        this._material.emissionColorTexture = value;
    }
    /**
     * Gets the emission color texture of the OpenPBR material.
     * @returns The emission texture or null
     */
    get emissionColorTexture() {
        return this._material.emissionColorTexture;
    }
    // ========================================
    // AMBIENT OCCLUSION
    // ========================================
    /**
     * Sets the ambient occlusion texture of the OpenPBR material.
     * @param value The ambient occlusion texture or null
     */
    set ambientOcclusionTexture(value) {
        this._material.ambientOcclusionTexture = value;
    }
    /**
     * Gets the ambient occlusion texture of the OpenPBR material.
     * @returns The ambient occlusion texture or null
     */
    get ambientOcclusionTexture() {
        return this._material.ambientOcclusionTexture;
    }
    /**
     * Sets the ambient occlusion texture strength by modifying the texture's level.
     * @param value The strength value (typically 0-1)
     */
    set ambientOcclusionTextureStrength(value) {
        const texture = this._material.ambientOcclusionTexture;
        if (texture) {
            texture.level = value;
        }
    }
    /**
     * Gets the ambient occlusion texture strength from the texture's level property.
     * @returns The strength value, defaults to 1.0 if no texture or level is set
     */
    get ambientOcclusionTextureStrength() {
        const texture = this._material.ambientOcclusionTexture;
        return texture?.level ?? 1.0;
    }
    // ========================================
    // COAT PARAMETERS
    // ========================================
    /**
     * Configures coat parameters for OpenPBR material.
     * OpenPBR coat is already built-in, so no configuration is needed.
     */
    configureCoat() {
        // OpenPBR coat is already built-in, no configuration needed
    }
    /**
     * Sets the coat weight of the OpenPBR material.
     * @param value The coat weight value (0-1)
     */
    set coatWeight(value) {
        this._material.coatWeight = value;
    }
    /**
     * Gets the coat weight of the OpenPBR material.
     * @returns The coat weight value (0-1)
     */
    get coatWeight() {
        return this._material.coatWeight;
    }
    /**
     * Sets the coat weight texture of the OpenPBR material.
     * @param value The coat weight texture or null
     */
    set coatWeightTexture(value) {
        this._material.coatWeightTexture = value;
    }
    /**
     * Gets the coat weight texture of the OpenPBR material.
     * @returns The coat weight texture or null
     */
    get coatWeightTexture() {
        return this._material.coatWeightTexture;
    }
    /**
     * Sets the coat color of the OpenPBR material.
     * @param value The coat color as a Color3
     */
    set coatColor(value) {
        this._material.coatColor = value;
    }
    /**
     * Sets the coat color texture of the OpenPBR material.
     * @param value The coat color texture or null
     */
    set coatColorTexture(value) {
        this._material.coatColorTexture = value;
    }
    /**
     * Sets the coat roughness of the OpenPBR material.
     * @param value The coat roughness value (0-1)
     */
    set coatRoughness(value) {
        this._material.coatRoughness = value;
    }
    /**
     * Gets the coat roughness of the OpenPBR material.
     * @returns The coat roughness value (0-1)
     */
    get coatRoughness() {
        return this._material.coatRoughness;
    }
    /**
     * Sets the coat roughness texture of the OpenPBR material.
     * @param value The coat roughness texture or null
     */
    set coatRoughnessTexture(value) {
        this._material.coatRoughnessTexture = value;
        if (value) {
            this._material._useCoatRoughnessFromGreenChannel = true;
        }
    }
    /**
     * Gets the coat roughness texture of the OpenPBR material.
     * @returns The coat roughness texture or null
     */
    get coatRoughnessTexture() {
        return this._material.coatRoughnessTexture;
    }
    /**
     * Sets the coat index of refraction (IOR) of the OpenPBR material.
     */
    set coatIor(value) {
        this._material.coatIor = value;
    }
    /**
     * Sets the coat darkening value of the OpenPBR material.
     * @param value The coat darkening value
     */
    set coatDarkening(value) {
        this._material.coatDarkening = value;
    }
    /**
     * Sets the coat darkening texture (OpenPBR: coatDarkeningTexture, no PBR equivalent)
     */
    set coatDarkeningTexture(value) {
        this._material.coatDarkeningTexture = value;
    }
    /**
     * Sets the coat roughness anisotropy.
     * TODO: Implementation pending OpenPBR coat anisotropy feature availability.
     * @param value The coat anisotropy intensity value
     */
    set coatRoughnessAnisotropy(value) {
        this._material.coatRoughnessAnisotropy = value;
    }
    /**
     * Gets the coat roughness anisotropy.
     * TODO: Implementation pending OpenPBR coat anisotropy feature availability.
     * @returns Currently returns 0 as coat anisotropy is not yet available
     */
    get coatRoughnessAnisotropy() {
        return this._material.coatRoughnessAnisotropy;
    }
    /**
     * Sets the coat tangent angle for anisotropy.
     * TODO: Implementation pending OpenPBR coat anisotropy feature availability.
     * @param value The coat anisotropy rotation angle in radians
     */
    set geometryCoatTangentAngle(value) {
        this._material.geometryCoatTangentAngle = value;
    }
    /**
     * Sets the coat tangent texture for anisotropy.
     * TODO: Implementation pending OpenPBR coat anisotropy feature availability.
     * @param value The coat anisotropy texture or null
     */
    set geometryCoatTangentTexture(value) {
        this._material.geometryCoatTangentTexture = value;
        if (value) {
            this._material._useCoatRoughnessAnisotropyFromTangentTexture = true;
        }
    }
    /**
     * Gets the coat tangent texture for anisotropy.
     * TODO: Implementation pending OpenPBR coat anisotropy feature availability.
     * @returns Currently returns null as coat anisotropy is not yet available
     */
    get geometryCoatTangentTexture() {
        return this._material.geometryCoatTangentTexture;
    }
    // ========================================
    // TRANSMISSION LAYER
    // ========================================
    /**
     * Sets the transmission weight.
     * TODO: Implementation pending OpenPBR transmission feature availability.
     * @param value The transmission weight value (0-1)
     */
    set transmissionWeight(value) {
        // TODO: Implement when OpenPBR transmission is available
        // this._material.transmissionWeight = value;
    }
    /**
     * Sets the transmission weight texture.
     * TODO: Implementation pending OpenPBR transmission feature availability.
     * @param value The transmission weight texture or null
     */
    set transmissionWeightTexture(value) {
        // TODO: Implement when OpenPBR transmission is available
        // this._material.transmissionWeightTexture = value;
    }
    /**
     * Gets the transmission weight.
     * TODO: Implementation pending OpenPBR transmission feature availability.
     * @returns Currently returns 0 as transmission is not yet available
     */
    get transmissionWeight() {
        // TODO: Implement when OpenPBR transmission is available
        // return this._material.transmissionWeight;
        return 0;
    }
    /**
     * Gets the transmission dispersion Abbe number.
     * @param value The Abbe number value
     */
    set transmissionDispersionAbbeNumber(value) {
        // TODO: Implement when OpenPBR transmission dispersion is available
    }
    /**
     * Configures transmission for OpenPBR material.
     * TODO: Implementation pending OpenPBR transmission feature availability.
     */
    configureTransmission() {
        // OpenPBR transmission will be configured differently when available
    }
    // ========================================
    // VOLUME PROPERTIES (Subsurface Scattering)
    // ========================================
    /**
     * Sets the attenuation distance for volume scattering.
     * TODO: Implementation pending OpenPBR volume feature availability.
     * @param value The attenuation distance value
     */
    set transmissionDepth(value) {
        // TODO: Implement when OpenPBR volume properties are available
        // this._material.attenuationDistance = value;
    }
    /**
     * Sets the attenuation color for volume scattering.
     * TODO: Implementation pending OpenPBR volume feature availability.
     * @param value The attenuation color as a Color3
     */
    set transmissionColor(value) {
        // TODO: Implement when OpenPBR volume properties are available
        // this._material.attenuationColor = value;
    }
    /**
     * Sets the thickness texture for volume scattering.
     * TODO: Implementation pending OpenPBR volume feature availability.
     * @param value The thickness texture or null
     */
    set volumeThicknessTexture(value) {
        // TODO: Implement when OpenPBR volume properties are available
        // this._material.thicknessTexture = value;
    }
    /**
     * Sets the thickness factor for volume scattering.
     * TODO: Implementation pending OpenPBR volume feature availability.
     * @param value The thickness value
     */
    set volumeThickness(value) {
        // TODO: Implement when OpenPBR volume properties are available
        // this._material.thickness = value;
    }
    // ========================================
    // SUBSURFACE PROPERTIES (Subsurface Scattering)
    // ========================================
    /**
     * Configures subsurface properties for PBR material
     */
    configureSubsurface() {
        // TODO
    }
    /**
     * Sets the subsurface weight
     */
    set subsurfaceWeight(value) {
        // TODO
    }
    get subsurfaceWeight() {
        // TODO
        return 0;
    }
    /**
     * Sets the subsurface weight texture
     */
    set subsurfaceWeightTexture(value) {
        // TODO
    }
    /**
     * Sets the subsurface color.
     * @param value The subsurface tint color as a Color3
     */
    set subsurfaceColor(value) {
        // TODO
    }
    /**
     * Sets the subsurface color texture.
     * @param value The subsurface tint texture or null
     */
    set subsurfaceColorTexture(value) {
        // TODO
    }
    // ========================================
    // FUZZ LAYER (Sheen)
    // ========================================
    /**
     * Configures fuzz for OpenPBR.
     * Enables fuzz and sets up proper configuration.
     */
    configureFuzz() {
        // Currently no setup to do for OpenPBR
    }
    /**
     * Sets the fuzz weight.
     * @param value The fuzz weight value
     */
    set fuzzWeight(value) {
        this._material.fuzzWeight = value;
    }
    /**
     * Sets the fuzz weight texture.
     * @param value The fuzz weight texture or null
     */
    set fuzzWeightTexture(value) {
        this._material.fuzzWeightTexture = value;
    }
    /**
     * Sets the fuzz color.
     * @param value The fuzz color as a Color3
     */
    set fuzzColor(value) {
        this._material.fuzzColor = value;
    }
    /**
     * Sets the fuzz color texture.
     * @param value The fuzz color texture or null
     */
    set fuzzColorTexture(value) {
        this._material.fuzzColorTexture = value;
    }
    /**
     * Sets the fuzz roughness.
     * @param value The fuzz roughness value (0-1)
     */
    set fuzzRoughness(value) {
        this._material.fuzzRoughness = value;
    }
    /**
     * Sets the fuzz roughness texture.
     * @param value The fuzz roughness texture or null
     */
    set fuzzRoughnessTexture(value) {
        this._material.fuzzRoughnessTexture = value;
        this._material._useFuzzRoughnessFromTextureAlpha = true;
    }
    // ========================================
    // ANISOTROPY
    // ========================================
    /**
     * Sets the specular roughness anisotropy of the OpenPBR material.
     * @param value The anisotropy intensity value
     */
    set specularRoughnessAnisotropy(value) {
        this._material.specularRoughnessAnisotropy = value;
    }
    /**
     * Gets the specular roughness anisotropy of the OpenPBR material.
     * @returns The anisotropy intensity value
     */
    get specularRoughnessAnisotropy() {
        return this._material.specularRoughnessAnisotropy;
    }
    /**
     * Sets the anisotropy rotation angle.
     * @param value The anisotropy rotation angle in radians
     */
    set geometryTangentAngle(value) {
        this._material.geometryTangentAngle = value;
    }
    /**
     * Sets the geometry tangent texture for anisotropy.
     * Automatically enables using anisotropy from the tangent texture.
     * @param value The anisotropy texture or null
     */
    set geometryTangentTexture(value) {
        this._material.geometryTangentTexture = value;
        this._material._useSpecularRoughnessAnisotropyFromTangentTexture = true;
    }
    /**
     * Gets the geometry tangent texture for anisotropy.
     * @returns The anisotropy texture or null
     */
    get geometryTangentTexture() {
        return this._material.geometryTangentTexture;
    }
    /**
     * Configures glTF-style anisotropy for the OpenPBR material.
     * @param useGltfStyle Whether to use glTF-style anisotropy
     */
    configureGltfStyleAnisotropy(useGltfStyle = true) {
        this._material._useGltfStyleAnisotropy = useGltfStyle;
    }
    // ========================================
    // THIN FILM IRIDESCENCE
    // ========================================
    /**
     * Sets the thin film weight.
     * @param value The thin film weight value
     */
    set thinFilmWeight(value) {
        this._material.thinFilmWeight = value;
    }
    /**
     * Sets the thin film IOR.
     * @param value The thin film IOR value
     */
    set thinFilmIor(value) {
        this._material.thinFilmIor = value;
    }
    /**
     * Sets the thin film thickness minimum.
     * @param value The minimum thickness value in nanometers
     */
    set thinFilmThicknessMinimum(value) {
        this._material.thinFilmThicknessMin = value / 1000.0; // Convert to micrometers for OpenPBR
    }
    /**
     * Sets the thin film thickness maximum.
     * @param value The maximum thickness value in nanometers
     */
    set thinFilmThicknessMaximum(value) {
        this._material.thinFilmThickness = value / 1000.0; // Convert to micrometers for OpenPBR
    }
    /**
     * Sets the thin film weight texture.
     * @param value The thin film weight texture or null
     */
    set thinFilmWeightTexture(value) {
        this._material.thinFilmWeightTexture = value;
    }
    /**
     * Sets the thin film thickness texture.
     * @param value The thin film thickness texture or null
     */
    set thinFilmThicknessTexture(value) {
        this._material.thinFilmThicknessTexture = value;
        this._material._useThinFilmThicknessFromTextureGreen = true;
    }
    // ========================================
    // UNLIT MATERIALS
    // ========================================
    /**
     * Sets whether the OpenPBR material is unlit.
     * @param value True to make the material unlit
     */
    set unlit(value) {
        this._material.unlit = value;
    }
    // ========================================
    // GEOMETRY PARAMETERS
    // ========================================
    /**
     * Sets the geometry opacity of the OpenPBR material.
     * @param value The opacity value (0-1)
     */
    set geometryOpacity(value) {
        this._material.geometryOpacity = value;
    }
    /**
     * Gets the geometry opacity of the OpenPBR material.
     * @returns The opacity value (0-1)
     */
    get geometryOpacity() {
        return this._material.geometryOpacity;
    }
    /**
     * Sets the geometry normal texture of the OpenPBR material.
     * @param value The normal texture or null
     */
    set geometryNormalTexture(value) {
        this._material.geometryNormalTexture = value;
    }
    /**
     * Gets the geometry normal texture of the OpenPBR material.
     * @returns The normal texture or null
     */
    get geometryNormalTexture() {
        return this._material.geometryNormalTexture;
    }
    /**
     * Sets the normal map inversions for the OpenPBR material.
     * Note: OpenPBR may handle normal map inversions differently or may not need them.
     * @param invertX Whether to invert the normal map on the X axis (may be ignored)
     * @param invertY Whether to invert the normal map on the Y axis (may be ignored)
     */
    setNormalMapInversions(invertX, invertY) {
        // OpenPBR handles normal map inversions differently or may not need them
    }
    /**
     * Sets the geometry coat normal texture of the OpenPBR material.
     * @param value The coat normal texture or null
     */
    set geometryCoatNormalTexture(value) {
        this._material.geometryCoatNormalTexture = value;
    }
    /**
     * Gets the geometry coat normal texture of the OpenPBR material.
     * @returns The coat normal texture or null
     */
    get geometryCoatNormalTexture() {
        return this._material.geometryCoatNormalTexture;
    }
    /**
     * Sets the geometry coat normal texture scale.
     * @param value The scale value for the coat normal texture
     */
    set geometryCoatNormalTextureScale(value) {
        if (this._material.geometryCoatNormalTexture) {
            this._material.geometryCoatNormalTexture.level = value;
        }
    }
}

export { OpenPBRMaterialLoadingAdapter };
//# sourceMappingURL=openpbrMaterialLoadingAdapter-Br2hIutA.js.map
