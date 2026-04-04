function GetByteIndex(bitIndex) {
    return Math.floor(bitIndex / 8);
}
function GetBitMask(bitIndex) {
    return 1 << bitIndex % 8;
}
/**
 * An fixed size array that effectively stores boolean values where each value is a single bit of backing data.
 * @remarks
 * All bits are initialized to false.
 */
class BitArray {
    /**
     * Creates a new bit array with a fixed size.
     * @param size The number of bits to store.
     */
    constructor(size) {
        this.size = size;
        this._byteArray = new Uint8Array(Math.ceil(this.size / 8));
    }
    /**
     * Gets the current value at the specified index.
     * @param bitIndex The index to get the value from.
     * @returns The value at the specified index.
     */
    get(bitIndex) {
        if (bitIndex >= this.size) {
            throw new RangeError("Bit index out of range");
        }
        const byteIndex = GetByteIndex(bitIndex);
        const bitMask = GetBitMask(bitIndex);
        return (this._byteArray[byteIndex] & bitMask) !== 0;
    }
    /**
     * Sets the value at the specified index.
     * @param bitIndex The index to set the value at.
     * @param value The value to set.
     */
    set(bitIndex, value) {
        if (bitIndex >= this.size) {
            throw new RangeError("Bit index out of range");
        }
        const byteIndex = GetByteIndex(bitIndex);
        const bitMask = GetBitMask(bitIndex);
        if (value) {
            this._byteArray[byteIndex] |= bitMask;
        }
        else {
            this._byteArray[byteIndex] &= ~bitMask;
        }
    }
}

/**
 * Sort (in place) the index array so that faces with common indices are close
 * @param indices the array of indices to sort
 */
function OptimizeIndices(indices) {
    const faces = [];
    const faceCount = indices.length / 3;
    // Step 1: Break the indices array into faces
    for (let i = 0; i < faceCount; i++) {
        faces.push([indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]]);
    }
    // Step 2: Build a graph connecting faces sharing a vertex
    const vertexToFaceMap = new Map();
    for (let faceIndex = 0; faceIndex < faces.length; faceIndex++) {
        const face = faces[faceIndex];
        for (const vertex of face) {
            let face = vertexToFaceMap.get(vertex);
            if (!face) {
                vertexToFaceMap.set(vertex, (face = []));
            }
            face.push(faceIndex);
        }
    }
    // Step 3: Traverse faces using DFS to ensure connected faces are close
    const visited = new BitArray(faceCount);
    const sortedFaces = [];
    // Using a stack and not a recursive version to avoid call stack overflow
    const deepFirstSearchStack = (startFaceIndex) => {
        const stack = [startFaceIndex];
        while (stack.length > 0) {
            const currentFaceIndex = stack.pop();
            if (visited.get(currentFaceIndex)) {
                continue;
            }
            visited.set(currentFaceIndex, true);
            sortedFaces.push(faces[currentFaceIndex]);
            // Push unvisited neighbors (faces sharing a vertex) onto the stack
            for (const vertex of faces[currentFaceIndex]) {
                const neighbors = vertexToFaceMap.get(vertex);
                if (!neighbors) {
                    return;
                }
                for (const neighborFaceIndex of neighbors) {
                    if (!visited.get(neighborFaceIndex)) {
                        stack.push(neighborFaceIndex);
                    }
                }
            }
        }
    };
    // Start DFS from the first face
    for (let i = 0; i < faceCount; i++) {
        if (!visited.get(i)) {
            deepFirstSearchStack(i);
        }
    }
    // Step 4: Flatten the sorted faces back into an array
    let index = 0;
    for (const face of sortedFaces) {
        indices[index++] = face[0];
        indices[index++] = face[1];
        indices[index++] = face[2];
    }
}

export { OptimizeIndices };
//# sourceMappingURL=mesh.vertexData.functions-D7sV26vR.js.map
