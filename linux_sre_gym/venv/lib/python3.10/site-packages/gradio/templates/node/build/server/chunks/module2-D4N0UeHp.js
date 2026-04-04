const createAddUniqueNumber = (generateUniqueNumber2) => {
  return (set) => {
    const number = generateUniqueNumber2(set);
    set.add(number);
    return number;
  };
};
const createCache = (lastNumberWeakMap) => {
  return (collection, nextNumber) => {
    lastNumberWeakMap.set(collection, nextNumber);
    return nextNumber;
  };
};
const MAX_SAFE_INTEGER = Number.MAX_SAFE_INTEGER === void 0 ? 9007199254740991 : Number.MAX_SAFE_INTEGER;
const TWO_TO_THE_POWER_OF_TWENTY_NINE = 536870912;
const TWO_TO_THE_POWER_OF_THIRTY = TWO_TO_THE_POWER_OF_TWENTY_NINE * 2;
const createGenerateUniqueNumber = (cache2, lastNumberWeakMap) => {
  return (collection) => {
    const lastNumber = lastNumberWeakMap.get(collection);
    let nextNumber = lastNumber === void 0 ? collection.size : lastNumber < TWO_TO_THE_POWER_OF_THIRTY ? lastNumber + 1 : 0;
    if (!collection.has(nextNumber)) {
      return cache2(collection, nextNumber);
    }
    if (collection.size < TWO_TO_THE_POWER_OF_TWENTY_NINE) {
      while (collection.has(nextNumber)) {
        nextNumber = Math.floor(Math.random() * TWO_TO_THE_POWER_OF_THIRTY);
      }
      return cache2(collection, nextNumber);
    }
    if (collection.size > MAX_SAFE_INTEGER) {
      throw new Error("Congratulations, you created a collection of unique numbers which uses all available integers!");
    }
    while (collection.has(nextNumber)) {
      nextNumber = Math.floor(Math.random() * MAX_SAFE_INTEGER);
    }
    return cache2(collection, nextNumber);
  };
};
const LAST_NUMBER_WEAK_MAP = /* @__PURE__ */ new WeakMap();
const cache = createCache(LAST_NUMBER_WEAK_MAP);
const generateUniqueNumber = createGenerateUniqueNumber(cache, LAST_NUMBER_WEAK_MAP);
const addUniqueNumber = createAddUniqueNumber(generateUniqueNumber);
const createBrokerFactory = (createOrGetOngoingRequests, extendBrokerImplementation, generateUniqueNumber2, isMessagePort2) => (brokerImplementation) => {
  const fullBrokerImplementation = extendBrokerImplementation(brokerImplementation);
  return (sender) => {
    const ongoingRequests = createOrGetOngoingRequests(sender);
    sender.addEventListener("message", (({ data: message }) => {
      const { id } = message;
      if (id !== null && ongoingRequests.has(id)) {
        const { reject, resolve } = ongoingRequests.get(id);
        ongoingRequests.delete(id);
        if (message.error === void 0) {
          resolve(message.result);
        } else {
          reject(new Error(message.error.message));
        }
      }
    }));
    if (isMessagePort2(sender)) {
      sender.start();
    }
    const call = (method, params = null, transferables = []) => {
      return new Promise((resolve, reject) => {
        const id = generateUniqueNumber2(ongoingRequests);
        ongoingRequests.set(id, { reject, resolve });
        if (params === null) {
          sender.postMessage({ id, method }, transferables);
        } else {
          sender.postMessage({ id, method, params }, transferables);
        }
      });
    };
    const notify = (method, params, transferables = []) => {
      sender.postMessage({ id: null, method, params }, transferables);
    };
    let functions = {};
    for (const [key, handler] of Object.entries(fullBrokerImplementation)) {
      functions = { ...functions, [key]: handler({ call, notify }) };
    }
    return { ...functions };
  };
};
const createCreateOrGetOngoingRequests = (ongoingRequestsMap) => (sender) => {
  if (ongoingRequestsMap.has(sender)) {
    return ongoingRequestsMap.get(sender);
  }
  const ongoingRequests = /* @__PURE__ */ new Map();
  ongoingRequestsMap.set(sender, ongoingRequests);
  return ongoingRequests;
};
const createExtendBrokerImplementation = (portMap) => (partialBrokerImplementation) => ({
  ...partialBrokerImplementation,
  connect: ({ call }) => {
    return async () => {
      const { port1, port2 } = new MessageChannel();
      const portId = await call("connect", { port: port1 }, [port1]);
      portMap.set(port2, portId);
      return port2;
    };
  },
  disconnect: ({ call }) => {
    return async (port) => {
      const portId = portMap.get(port);
      if (portId === void 0) {
        throw new Error("The given port is not connected.");
      }
      await call("disconnect", { portId });
    };
  },
  isSupported: ({ call }) => {
    return () => call("isSupported");
  }
});
const isMessagePort = (sender) => {
  return typeof sender.start === "function";
};
const createBroker = createBrokerFactory(createCreateOrGetOngoingRequests(/* @__PURE__ */ new WeakMap()), createExtendBrokerImplementation(/* @__PURE__ */ new WeakMap()), generateUniqueNumber, isMessagePort);

export { addUniqueNumber as a, createBroker as c, generateUniqueNumber as g };
//# sourceMappingURL=module2-D4N0UeHp.js.map
