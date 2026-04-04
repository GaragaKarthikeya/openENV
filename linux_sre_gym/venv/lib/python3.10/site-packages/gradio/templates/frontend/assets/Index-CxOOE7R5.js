import { p as prop, b as set_class, a as set_attribute, h as clsx, i as if_block, d as bind_this, f as set_style, t as remove_input_defaults, F as set_value, k as each, u as index, g as spread_props, r as rest_props } from './i18n-CMo5lpzy.js';
import { u as state, v as proxy, ai as user_pre_effect, w as get, x as set, y as user_effect, z as untrack, aj as delegate, R as push, t as template_effect, a as append, T as pop, a5 as user_derived, V as child, W as from_html, Y as reset, S as first_child, X as sibling, A as effect, Z as event, ak as remove_textarea_child, a6 as comment, a7 as text$1, a0 as set_text, f as from_svg, aa as onMount, M as tick, aw as $window } from './index-BvBk1Iap.js';
import { h as html } from './html-CdUEtR5E.js';
import { a as action } from './actions-Cit_UA5S.js';
import { U as Upload } from './Upload-CPeZiqam.js';
import { c as component, I as IconButton } from './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { b as bind_value } from './input-C317uaXR.js';
import { M as MarkdownCode } from './MarkdownCode-Cg7CjFvQ.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import { C as Checkbox } from './Checkbox-BwK0lQJ8.js';
import { C as Check } from './Check-BlnBgjLT.js';
import { D as DropdownArrow } from './DropdownArrow-Dw1NZKCM.js';
import { C as Copy } from './Copy-DFFray7x.js';
import { F as FullscreenButton } from './FullscreenButton-CNmhk6nd.js';
import { d as dsvFormat } from './dsv-BhAd467f.js';
import { S as Static } from './index-Lz7fsHFt.js';
import { B as Block } from './Block-Bx3BikCN.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
export { default as BaseExample } from './Example-DL_4juBu.js';
import './snippet-Bl5YVoCg.js';
import './prism-python-CbXMXbdR.js';
import './index-C7e-J7CF.js';
import './clone-DaB-S2nH.js';
import './Maximize-B_43ylTs.js';
import './Clear-Dgs5UhTO.js';

/**
   * table-core
   *
   * Copyright (c) TanStack
   *
   * This source code is licensed under the MIT license found in the
   * LICENSE.md file in the root directory of this source tree.
   *
   * @license MIT
   */
function functionalUpdate(updater, input) {
  return typeof updater === "function" ? updater(input) : updater;
}
function makeStateUpdater(key, instance) {
  return (updater) => {
    instance.setState((old) => {
      return {
        ...old,
        [key]: functionalUpdate(updater, old[key])
      };
    });
  };
}
function isFunction(d) {
  return d instanceof Function;
}
function isNumberArray(d) {
  return Array.isArray(d) && d.every((val) => typeof val === "number");
}
function flattenBy(arr, getChildren) {
  const flat = [];
  const recurse = (subArr) => {
    subArr.forEach((item) => {
      flat.push(item);
      const children = getChildren(item);
      if (children != null && children.length) {
        recurse(children);
      }
    });
  };
  recurse(arr);
  return flat;
}
function memo$1(getDeps, fn, opts) {
  let deps = [];
  let result;
  return (depArgs) => {
    let depTime;
    if (opts.key && opts.debug) depTime = Date.now();
    const newDeps = getDeps(depArgs);
    const depsChanged = newDeps.length !== deps.length || newDeps.some((dep, index) => deps[index] !== dep);
    if (!depsChanged) {
      return result;
    }
    deps = newDeps;
    let resultTime;
    if (opts.key && opts.debug) resultTime = Date.now();
    result = fn(...newDeps);
    opts == null || opts.onChange == null || opts.onChange(result);
    if (opts.key && opts.debug) {
      if (opts != null && opts.debug()) {
        const depEndTime = Math.round((Date.now() - depTime) * 100) / 100;
        const resultEndTime = Math.round((Date.now() - resultTime) * 100) / 100;
        const resultFpsPercentage = resultEndTime / 16;
        const pad = (str, num) => {
          str = String(str);
          while (str.length < num) {
            str = " " + str;
          }
          return str;
        };
        console.info(`%c⏱ ${pad(resultEndTime, 5)} /${pad(depEndTime, 5)} ms`, `
            font-size: .6rem;
            font-weight: bold;
            color: hsl(${Math.max(0, Math.min(120 - 120 * resultFpsPercentage, 120))}deg 100% 31%);`, opts == null ? void 0 : opts.key);
      }
    }
    return result;
  };
}
function getMemoOptions(tableOptions, debugLevel, key, onChange) {
  return {
    debug: () => {
      var _tableOptions$debugAl;
      return (_tableOptions$debugAl = tableOptions == null ? void 0 : tableOptions.debugAll) != null ? _tableOptions$debugAl : tableOptions[debugLevel];
    },
    key: false,
    onChange
  };
}
function createCell(table, row, column, columnId) {
  const getRenderValue = () => {
    var _cell$getValue;
    return (_cell$getValue = cell.getValue()) != null ? _cell$getValue : table.options.renderFallbackValue;
  };
  const cell = {
    id: `${row.id}_${column.id}`,
    row,
    column,
    getValue: () => row.getValue(columnId),
    renderValue: getRenderValue,
    getContext: memo$1(() => [table, column, row, cell], (table2, column2, row2, cell2) => ({
      table: table2,
      column: column2,
      row: row2,
      cell: cell2,
      getValue: cell2.getValue,
      renderValue: cell2.renderValue
    }), getMemoOptions(table.options, "debugCells"))
  };
  table._features.forEach((feature) => {
    feature.createCell == null || feature.createCell(cell, column, row, table);
  }, {});
  return cell;
}
function createColumn(table, columnDef, depth, parent) {
  var _ref, _resolvedColumnDef$id;
  const defaultColumn = table._getDefaultColumnDef();
  const resolvedColumnDef = {
    ...defaultColumn,
    ...columnDef
  };
  const accessorKey = resolvedColumnDef.accessorKey;
  let id = (_ref = (_resolvedColumnDef$id = resolvedColumnDef.id) != null ? _resolvedColumnDef$id : accessorKey ? typeof String.prototype.replaceAll === "function" ? accessorKey.replaceAll(".", "_") : accessorKey.replace(/\./g, "_") : void 0) != null ? _ref : typeof resolvedColumnDef.header === "string" ? resolvedColumnDef.header : void 0;
  let accessorFn;
  if (resolvedColumnDef.accessorFn) {
    accessorFn = resolvedColumnDef.accessorFn;
  } else if (accessorKey) {
    if (accessorKey.includes(".")) {
      accessorFn = (originalRow) => {
        let result = originalRow;
        for (const key of accessorKey.split(".")) {
          var _result;
          result = (_result = result) == null ? void 0 : _result[key];
        }
        return result;
      };
    } else {
      accessorFn = (originalRow) => originalRow[resolvedColumnDef.accessorKey];
    }
  }
  if (!id) {
    throw new Error();
  }
  let column = {
    id: `${String(id)}`,
    accessorFn,
    parent,
    depth,
    columnDef: resolvedColumnDef,
    columns: [],
    getFlatColumns: memo$1(() => [true], () => {
      var _column$columns;
      return [column, ...(_column$columns = column.columns) == null ? void 0 : _column$columns.flatMap((d) => d.getFlatColumns())];
    }, getMemoOptions(table.options, "debugColumns")),
    getLeafColumns: memo$1(() => [table._getOrderColumnsFn()], (orderColumns2) => {
      var _column$columns2;
      if ((_column$columns2 = column.columns) != null && _column$columns2.length) {
        let leafColumns = column.columns.flatMap((column2) => column2.getLeafColumns());
        return orderColumns2(leafColumns);
      }
      return [column];
    }, getMemoOptions(table.options, "debugColumns"))
  };
  for (const feature of table._features) {
    feature.createColumn == null || feature.createColumn(column, table);
  }
  return column;
}
const debug = "debugHeaders";
function createHeader(table, column, options) {
  var _options$id;
  const id = (_options$id = options.id) != null ? _options$id : column.id;
  let header = {
    id,
    column,
    index: options.index,
    isPlaceholder: !!options.isPlaceholder,
    placeholderId: options.placeholderId,
    depth: options.depth,
    subHeaders: [],
    colSpan: 0,
    rowSpan: 0,
    headerGroup: null,
    getLeafHeaders: () => {
      const leafHeaders = [];
      const recurseHeader = (h) => {
        if (h.subHeaders && h.subHeaders.length) {
          h.subHeaders.map(recurseHeader);
        }
        leafHeaders.push(h);
      };
      recurseHeader(header);
      return leafHeaders;
    },
    getContext: () => ({
      table,
      header,
      column
    })
  };
  table._features.forEach((feature) => {
    feature.createHeader == null || feature.createHeader(header, table);
  });
  return header;
}
const Headers = {
  createTable: (table) => {
    table.getHeaderGroups = memo$1(() => [table.getAllColumns(), table.getVisibleLeafColumns(), table.getState().columnPinning.left, table.getState().columnPinning.right], (allColumns, leafColumns, left, right) => {
      var _left$map$filter, _right$map$filter;
      const leftColumns = (_left$map$filter = left == null ? void 0 : left.map((columnId) => leafColumns.find((d) => d.id === columnId)).filter(Boolean)) != null ? _left$map$filter : [];
      const rightColumns = (_right$map$filter = right == null ? void 0 : right.map((columnId) => leafColumns.find((d) => d.id === columnId)).filter(Boolean)) != null ? _right$map$filter : [];
      const centerColumns = leafColumns.filter((column) => !(left != null && left.includes(column.id)) && !(right != null && right.includes(column.id)));
      const headerGroups = buildHeaderGroups(allColumns, [...leftColumns, ...centerColumns, ...rightColumns], table);
      return headerGroups;
    }, getMemoOptions(table.options, debug));
    table.getCenterHeaderGroups = memo$1(() => [table.getAllColumns(), table.getVisibleLeafColumns(), table.getState().columnPinning.left, table.getState().columnPinning.right], (allColumns, leafColumns, left, right) => {
      leafColumns = leafColumns.filter((column) => !(left != null && left.includes(column.id)) && !(right != null && right.includes(column.id)));
      return buildHeaderGroups(allColumns, leafColumns, table, "center");
    }, getMemoOptions(table.options, debug));
    table.getLeftHeaderGroups = memo$1(() => [table.getAllColumns(), table.getVisibleLeafColumns(), table.getState().columnPinning.left], (allColumns, leafColumns, left) => {
      var _left$map$filter2;
      const orderedLeafColumns = (_left$map$filter2 = left == null ? void 0 : left.map((columnId) => leafColumns.find((d) => d.id === columnId)).filter(Boolean)) != null ? _left$map$filter2 : [];
      return buildHeaderGroups(allColumns, orderedLeafColumns, table, "left");
    }, getMemoOptions(table.options, debug));
    table.getRightHeaderGroups = memo$1(() => [table.getAllColumns(), table.getVisibleLeafColumns(), table.getState().columnPinning.right], (allColumns, leafColumns, right) => {
      var _right$map$filter2;
      const orderedLeafColumns = (_right$map$filter2 = right == null ? void 0 : right.map((columnId) => leafColumns.find((d) => d.id === columnId)).filter(Boolean)) != null ? _right$map$filter2 : [];
      return buildHeaderGroups(allColumns, orderedLeafColumns, table, "right");
    }, getMemoOptions(table.options, debug));
    table.getFooterGroups = memo$1(() => [table.getHeaderGroups()], (headerGroups) => {
      return [...headerGroups].reverse();
    }, getMemoOptions(table.options, debug));
    table.getLeftFooterGroups = memo$1(() => [table.getLeftHeaderGroups()], (headerGroups) => {
      return [...headerGroups].reverse();
    }, getMemoOptions(table.options, debug));
    table.getCenterFooterGroups = memo$1(() => [table.getCenterHeaderGroups()], (headerGroups) => {
      return [...headerGroups].reverse();
    }, getMemoOptions(table.options, debug));
    table.getRightFooterGroups = memo$1(() => [table.getRightHeaderGroups()], (headerGroups) => {
      return [...headerGroups].reverse();
    }, getMemoOptions(table.options, debug));
    table.getFlatHeaders = memo$1(() => [table.getHeaderGroups()], (headerGroups) => {
      return headerGroups.map((headerGroup) => {
        return headerGroup.headers;
      }).flat();
    }, getMemoOptions(table.options, debug));
    table.getLeftFlatHeaders = memo$1(() => [table.getLeftHeaderGroups()], (left) => {
      return left.map((headerGroup) => {
        return headerGroup.headers;
      }).flat();
    }, getMemoOptions(table.options, debug));
    table.getCenterFlatHeaders = memo$1(() => [table.getCenterHeaderGroups()], (left) => {
      return left.map((headerGroup) => {
        return headerGroup.headers;
      }).flat();
    }, getMemoOptions(table.options, debug));
    table.getRightFlatHeaders = memo$1(() => [table.getRightHeaderGroups()], (left) => {
      return left.map((headerGroup) => {
        return headerGroup.headers;
      }).flat();
    }, getMemoOptions(table.options, debug));
    table.getCenterLeafHeaders = memo$1(() => [table.getCenterFlatHeaders()], (flatHeaders) => {
      return flatHeaders.filter((header) => {
        var _header$subHeaders;
        return !((_header$subHeaders = header.subHeaders) != null && _header$subHeaders.length);
      });
    }, getMemoOptions(table.options, debug));
    table.getLeftLeafHeaders = memo$1(() => [table.getLeftFlatHeaders()], (flatHeaders) => {
      return flatHeaders.filter((header) => {
        var _header$subHeaders2;
        return !((_header$subHeaders2 = header.subHeaders) != null && _header$subHeaders2.length);
      });
    }, getMemoOptions(table.options, debug));
    table.getRightLeafHeaders = memo$1(() => [table.getRightFlatHeaders()], (flatHeaders) => {
      return flatHeaders.filter((header) => {
        var _header$subHeaders3;
        return !((_header$subHeaders3 = header.subHeaders) != null && _header$subHeaders3.length);
      });
    }, getMemoOptions(table.options, debug));
    table.getLeafHeaders = memo$1(() => [table.getLeftHeaderGroups(), table.getCenterHeaderGroups(), table.getRightHeaderGroups()], (left, center, right) => {
      var _left$0$headers, _left$, _center$0$headers, _center$, _right$0$headers, _right$;
      return [...(_left$0$headers = (_left$ = left[0]) == null ? void 0 : _left$.headers) != null ? _left$0$headers : [], ...(_center$0$headers = (_center$ = center[0]) == null ? void 0 : _center$.headers) != null ? _center$0$headers : [], ...(_right$0$headers = (_right$ = right[0]) == null ? void 0 : _right$.headers) != null ? _right$0$headers : []].map((header) => {
        return header.getLeafHeaders();
      }).flat();
    }, getMemoOptions(table.options, debug));
  }
};
function buildHeaderGroups(allColumns, columnsToGroup, table, headerFamily) {
  var _headerGroups$0$heade, _headerGroups$;
  let maxDepth = 0;
  const findMaxDepth = function(columns, depth) {
    if (depth === void 0) {
      depth = 1;
    }
    maxDepth = Math.max(maxDepth, depth);
    columns.filter((column) => column.getIsVisible()).forEach((column) => {
      var _column$columns;
      if ((_column$columns = column.columns) != null && _column$columns.length) {
        findMaxDepth(column.columns, depth + 1);
      }
    }, 0);
  };
  findMaxDepth(allColumns);
  let headerGroups = [];
  const createHeaderGroup = (headersToGroup, depth) => {
    const headerGroup = {
      depth,
      id: [headerFamily, `${depth}`].filter(Boolean).join("_"),
      headers: []
    };
    const pendingParentHeaders = [];
    headersToGroup.forEach((headerToGroup) => {
      const latestPendingParentHeader = [...pendingParentHeaders].reverse()[0];
      const isLeafHeader = headerToGroup.column.depth === headerGroup.depth;
      let column;
      let isPlaceholder = false;
      if (isLeafHeader && headerToGroup.column.parent) {
        column = headerToGroup.column.parent;
      } else {
        column = headerToGroup.column;
        isPlaceholder = true;
      }
      if (latestPendingParentHeader && (latestPendingParentHeader == null ? void 0 : latestPendingParentHeader.column) === column) {
        latestPendingParentHeader.subHeaders.push(headerToGroup);
      } else {
        const header = createHeader(table, column, {
          id: [headerFamily, depth, column.id, headerToGroup == null ? void 0 : headerToGroup.id].filter(Boolean).join("_"),
          isPlaceholder,
          placeholderId: isPlaceholder ? `${pendingParentHeaders.filter((d) => d.column === column).length}` : void 0,
          depth,
          index: pendingParentHeaders.length
        });
        header.subHeaders.push(headerToGroup);
        pendingParentHeaders.push(header);
      }
      headerGroup.headers.push(headerToGroup);
      headerToGroup.headerGroup = headerGroup;
    });
    headerGroups.push(headerGroup);
    if (depth > 0) {
      createHeaderGroup(pendingParentHeaders, depth - 1);
    }
  };
  const bottomHeaders = columnsToGroup.map((column, index) => createHeader(table, column, {
    depth: maxDepth,
    index
  }));
  createHeaderGroup(bottomHeaders, maxDepth - 1);
  headerGroups.reverse();
  const recurseHeadersForSpans = (headers) => {
    const filteredHeaders = headers.filter((header) => header.column.getIsVisible());
    return filteredHeaders.map((header) => {
      let colSpan = 0;
      let rowSpan = 0;
      let childRowSpans = [0];
      if (header.subHeaders && header.subHeaders.length) {
        childRowSpans = [];
        recurseHeadersForSpans(header.subHeaders).forEach((_ref) => {
          let {
            colSpan: childColSpan,
            rowSpan: childRowSpan
          } = _ref;
          colSpan += childColSpan;
          childRowSpans.push(childRowSpan);
        });
      } else {
        colSpan = 1;
      }
      const minChildRowSpan = Math.min(...childRowSpans);
      rowSpan = rowSpan + minChildRowSpan;
      header.colSpan = colSpan;
      header.rowSpan = rowSpan;
      return {
        colSpan,
        rowSpan
      };
    });
  };
  recurseHeadersForSpans((_headerGroups$0$heade = (_headerGroups$ = headerGroups[0]) == null ? void 0 : _headerGroups$.headers) != null ? _headerGroups$0$heade : []);
  return headerGroups;
}
const createRow = (table, id, original, rowIndex, depth, subRows, parentId) => {
  let row = {
    id,
    index: rowIndex,
    original,
    depth,
    parentId,
    _valuesCache: {},
    _uniqueValuesCache: {},
    getValue: (columnId) => {
      if (row._valuesCache.hasOwnProperty(columnId)) {
        return row._valuesCache[columnId];
      }
      const column = table.getColumn(columnId);
      if (!(column != null && column.accessorFn)) {
        return void 0;
      }
      row._valuesCache[columnId] = column.accessorFn(row.original, rowIndex);
      return row._valuesCache[columnId];
    },
    getUniqueValues: (columnId) => {
      if (row._uniqueValuesCache.hasOwnProperty(columnId)) {
        return row._uniqueValuesCache[columnId];
      }
      const column = table.getColumn(columnId);
      if (!(column != null && column.accessorFn)) {
        return void 0;
      }
      if (!column.columnDef.getUniqueValues) {
        row._uniqueValuesCache[columnId] = [row.getValue(columnId)];
        return row._uniqueValuesCache[columnId];
      }
      row._uniqueValuesCache[columnId] = column.columnDef.getUniqueValues(row.original, rowIndex);
      return row._uniqueValuesCache[columnId];
    },
    renderValue: (columnId) => {
      var _row$getValue;
      return (_row$getValue = row.getValue(columnId)) != null ? _row$getValue : table.options.renderFallbackValue;
    },
    subRows: [],
    getLeafRows: () => flattenBy(row.subRows, (d) => d.subRows),
    getParentRow: () => row.parentId ? table.getRow(row.parentId, true) : void 0,
    getParentRows: () => {
      let parentRows = [];
      let currentRow = row;
      while (true) {
        const parentRow = currentRow.getParentRow();
        if (!parentRow) break;
        parentRows.push(parentRow);
        currentRow = parentRow;
      }
      return parentRows.reverse();
    },
    getAllCells: memo$1(() => [table.getAllLeafColumns()], (leafColumns) => {
      return leafColumns.map((column) => {
        return createCell(table, row, column, column.id);
      });
    }, getMemoOptions(table.options, "debugRows")),
    _getAllCellsByColumnId: memo$1(() => [row.getAllCells()], (allCells) => {
      return allCells.reduce((acc, cell) => {
        acc[cell.column.id] = cell;
        return acc;
      }, {});
    }, getMemoOptions(table.options, "debugRows"))
  };
  for (let i = 0; i < table._features.length; i++) {
    const feature = table._features[i];
    feature == null || feature.createRow == null || feature.createRow(row, table);
  }
  return row;
};
const ColumnFaceting = {
  createColumn: (column, table) => {
    column._getFacetedRowModel = table.options.getFacetedRowModel && table.options.getFacetedRowModel(table, column.id);
    column.getFacetedRowModel = () => {
      if (!column._getFacetedRowModel) {
        return table.getPreFilteredRowModel();
      }
      return column._getFacetedRowModel();
    };
    column._getFacetedUniqueValues = table.options.getFacetedUniqueValues && table.options.getFacetedUniqueValues(table, column.id);
    column.getFacetedUniqueValues = () => {
      if (!column._getFacetedUniqueValues) {
        return /* @__PURE__ */ new Map();
      }
      return column._getFacetedUniqueValues();
    };
    column._getFacetedMinMaxValues = table.options.getFacetedMinMaxValues && table.options.getFacetedMinMaxValues(table, column.id);
    column.getFacetedMinMaxValues = () => {
      if (!column._getFacetedMinMaxValues) {
        return void 0;
      }
      return column._getFacetedMinMaxValues();
    };
  }
};
const includesString = (row, columnId, filterValue) => {
  var _filterValue$toString, _row$getValue;
  const search = filterValue == null || (_filterValue$toString = filterValue.toString()) == null ? void 0 : _filterValue$toString.toLowerCase();
  return Boolean((_row$getValue = row.getValue(columnId)) == null || (_row$getValue = _row$getValue.toString()) == null || (_row$getValue = _row$getValue.toLowerCase()) == null ? void 0 : _row$getValue.includes(search));
};
includesString.autoRemove = (val) => testFalsey(val);
const includesStringSensitive = (row, columnId, filterValue) => {
  var _row$getValue2;
  return Boolean((_row$getValue2 = row.getValue(columnId)) == null || (_row$getValue2 = _row$getValue2.toString()) == null ? void 0 : _row$getValue2.includes(filterValue));
};
includesStringSensitive.autoRemove = (val) => testFalsey(val);
const equalsString = (row, columnId, filterValue) => {
  var _row$getValue3;
  return ((_row$getValue3 = row.getValue(columnId)) == null || (_row$getValue3 = _row$getValue3.toString()) == null ? void 0 : _row$getValue3.toLowerCase()) === (filterValue == null ? void 0 : filterValue.toLowerCase());
};
equalsString.autoRemove = (val) => testFalsey(val);
const arrIncludes = (row, columnId, filterValue) => {
  var _row$getValue4;
  return (_row$getValue4 = row.getValue(columnId)) == null ? void 0 : _row$getValue4.includes(filterValue);
};
arrIncludes.autoRemove = (val) => testFalsey(val);
const arrIncludesAll = (row, columnId, filterValue) => {
  return !filterValue.some((val) => {
    var _row$getValue5;
    return !((_row$getValue5 = row.getValue(columnId)) != null && _row$getValue5.includes(val));
  });
};
arrIncludesAll.autoRemove = (val) => testFalsey(val) || !(val != null && val.length);
const arrIncludesSome = (row, columnId, filterValue) => {
  return filterValue.some((val) => {
    var _row$getValue6;
    return (_row$getValue6 = row.getValue(columnId)) == null ? void 0 : _row$getValue6.includes(val);
  });
};
arrIncludesSome.autoRemove = (val) => testFalsey(val) || !(val != null && val.length);
const equals = (row, columnId, filterValue) => {
  return row.getValue(columnId) === filterValue;
};
equals.autoRemove = (val) => testFalsey(val);
const weakEquals = (row, columnId, filterValue) => {
  return row.getValue(columnId) == filterValue;
};
weakEquals.autoRemove = (val) => testFalsey(val);
const inNumberRange = (row, columnId, filterValue) => {
  let [min2, max2] = filterValue;
  const rowValue = row.getValue(columnId);
  return rowValue >= min2 && rowValue <= max2;
};
inNumberRange.resolveFilterValue = (val) => {
  let [unsafeMin, unsafeMax] = val;
  let parsedMin = typeof unsafeMin !== "number" ? parseFloat(unsafeMin) : unsafeMin;
  let parsedMax = typeof unsafeMax !== "number" ? parseFloat(unsafeMax) : unsafeMax;
  let min2 = unsafeMin === null || Number.isNaN(parsedMin) ? -Infinity : parsedMin;
  let max2 = unsafeMax === null || Number.isNaN(parsedMax) ? Infinity : parsedMax;
  if (min2 > max2) {
    const temp = min2;
    min2 = max2;
    max2 = temp;
  }
  return [min2, max2];
};
inNumberRange.autoRemove = (val) => testFalsey(val) || testFalsey(val[0]) && testFalsey(val[1]);
const filterFns = {
  includesString,
  includesStringSensitive,
  equalsString,
  arrIncludes,
  arrIncludesAll,
  arrIncludesSome,
  equals,
  weakEquals,
  inNumberRange
};
function testFalsey(val) {
  return val === void 0 || val === null || val === "";
}
const ColumnFiltering = {
  getDefaultColumnDef: () => {
    return {
      filterFn: "auto"
    };
  },
  getInitialState: (state) => {
    return {
      columnFilters: [],
      ...state
    };
  },
  getDefaultOptions: (table) => {
    return {
      onColumnFiltersChange: makeStateUpdater("columnFilters", table),
      filterFromLeafRows: false,
      maxLeafRowFilterDepth: 100
    };
  },
  createColumn: (column, table) => {
    column.getAutoFilterFn = () => {
      const firstRow = table.getCoreRowModel().flatRows[0];
      const value = firstRow == null ? void 0 : firstRow.getValue(column.id);
      if (typeof value === "string") {
        return filterFns.includesString;
      }
      if (typeof value === "number") {
        return filterFns.inNumberRange;
      }
      if (typeof value === "boolean") {
        return filterFns.equals;
      }
      if (value !== null && typeof value === "object") {
        return filterFns.equals;
      }
      if (Array.isArray(value)) {
        return filterFns.arrIncludes;
      }
      return filterFns.weakEquals;
    };
    column.getFilterFn = () => {
      var _table$options$filter, _table$options$filter2;
      return isFunction(column.columnDef.filterFn) ? column.columnDef.filterFn : column.columnDef.filterFn === "auto" ? column.getAutoFilterFn() : (
        // @ts-ignore
        (_table$options$filter = (_table$options$filter2 = table.options.filterFns) == null ? void 0 : _table$options$filter2[column.columnDef.filterFn]) != null ? _table$options$filter : filterFns[column.columnDef.filterFn]
      );
    };
    column.getCanFilter = () => {
      var _column$columnDef$ena, _table$options$enable, _table$options$enable2;
      return ((_column$columnDef$ena = column.columnDef.enableColumnFilter) != null ? _column$columnDef$ena : true) && ((_table$options$enable = table.options.enableColumnFilters) != null ? _table$options$enable : true) && ((_table$options$enable2 = table.options.enableFilters) != null ? _table$options$enable2 : true) && !!column.accessorFn;
    };
    column.getIsFiltered = () => column.getFilterIndex() > -1;
    column.getFilterValue = () => {
      var _table$getState$colum;
      return (_table$getState$colum = table.getState().columnFilters) == null || (_table$getState$colum = _table$getState$colum.find((d) => d.id === column.id)) == null ? void 0 : _table$getState$colum.value;
    };
    column.getFilterIndex = () => {
      var _table$getState$colum2, _table$getState$colum3;
      return (_table$getState$colum2 = (_table$getState$colum3 = table.getState().columnFilters) == null ? void 0 : _table$getState$colum3.findIndex((d) => d.id === column.id)) != null ? _table$getState$colum2 : -1;
    };
    column.setFilterValue = (value) => {
      table.setColumnFilters((old) => {
        const filterFn = column.getFilterFn();
        const previousFilter = old == null ? void 0 : old.find((d) => d.id === column.id);
        const newFilter = functionalUpdate(value, previousFilter ? previousFilter.value : void 0);
        if (shouldAutoRemoveFilter(filterFn, newFilter, column)) {
          var _old$filter;
          return (_old$filter = old == null ? void 0 : old.filter((d) => d.id !== column.id)) != null ? _old$filter : [];
        }
        const newFilterObj = {
          id: column.id,
          value: newFilter
        };
        if (previousFilter) {
          var _old$map;
          return (_old$map = old == null ? void 0 : old.map((d) => {
            if (d.id === column.id) {
              return newFilterObj;
            }
            return d;
          })) != null ? _old$map : [];
        }
        if (old != null && old.length) {
          return [...old, newFilterObj];
        }
        return [newFilterObj];
      });
    };
  },
  createRow: (row, _table) => {
    row.columnFilters = {};
    row.columnFiltersMeta = {};
  },
  createTable: (table) => {
    table.setColumnFilters = (updater) => {
      const leafColumns = table.getAllLeafColumns();
      const updateFn = (old) => {
        var _functionalUpdate;
        return (_functionalUpdate = functionalUpdate(updater, old)) == null ? void 0 : _functionalUpdate.filter((filter) => {
          const column = leafColumns.find((d) => d.id === filter.id);
          if (column) {
            const filterFn = column.getFilterFn();
            if (shouldAutoRemoveFilter(filterFn, filter.value, column)) {
              return false;
            }
          }
          return true;
        });
      };
      table.options.onColumnFiltersChange == null || table.options.onColumnFiltersChange(updateFn);
    };
    table.resetColumnFilters = (defaultState) => {
      var _table$initialState$c, _table$initialState;
      table.setColumnFilters(defaultState ? [] : (_table$initialState$c = (_table$initialState = table.initialState) == null ? void 0 : _table$initialState.columnFilters) != null ? _table$initialState$c : []);
    };
    table.getPreFilteredRowModel = () => table.getCoreRowModel();
    table.getFilteredRowModel = () => {
      if (!table._getFilteredRowModel && table.options.getFilteredRowModel) {
        table._getFilteredRowModel = table.options.getFilteredRowModel(table);
      }
      if (table.options.manualFiltering || !table._getFilteredRowModel) {
        return table.getPreFilteredRowModel();
      }
      return table._getFilteredRowModel();
    };
  }
};
function shouldAutoRemoveFilter(filterFn, value, column) {
  return (filterFn && filterFn.autoRemove ? filterFn.autoRemove(value, column) : false) || typeof value === "undefined" || typeof value === "string" && !value;
}
const sum = (columnId, _leafRows, childRows) => {
  return childRows.reduce((sum2, next) => {
    const nextValue = next.getValue(columnId);
    return sum2 + (typeof nextValue === "number" ? nextValue : 0);
  }, 0);
};
const min = (columnId, _leafRows, childRows) => {
  let min2;
  childRows.forEach((row) => {
    const value = row.getValue(columnId);
    if (value != null && (min2 > value || min2 === void 0 && value >= value)) {
      min2 = value;
    }
  });
  return min2;
};
const max = (columnId, _leafRows, childRows) => {
  let max2;
  childRows.forEach((row) => {
    const value = row.getValue(columnId);
    if (value != null && (max2 < value || max2 === void 0 && value >= value)) {
      max2 = value;
    }
  });
  return max2;
};
const extent = (columnId, _leafRows, childRows) => {
  let min2;
  let max2;
  childRows.forEach((row) => {
    const value = row.getValue(columnId);
    if (value != null) {
      if (min2 === void 0) {
        if (value >= value) min2 = max2 = value;
      } else {
        if (min2 > value) min2 = value;
        if (max2 < value) max2 = value;
      }
    }
  });
  return [min2, max2];
};
const mean = (columnId, leafRows) => {
  let count2 = 0;
  let sum2 = 0;
  leafRows.forEach((row) => {
    let value = row.getValue(columnId);
    if (value != null && (value = +value) >= value) {
      ++count2, sum2 += value;
    }
  });
  if (count2) return sum2 / count2;
  return;
};
const median = (columnId, leafRows) => {
  if (!leafRows.length) {
    return;
  }
  const values = leafRows.map((row) => row.getValue(columnId));
  if (!isNumberArray(values)) {
    return;
  }
  if (values.length === 1) {
    return values[0];
  }
  const mid = Math.floor(values.length / 2);
  const nums = values.sort((a, b) => a - b);
  return values.length % 2 !== 0 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2;
};
const unique = (columnId, leafRows) => {
  return Array.from(new Set(leafRows.map((d) => d.getValue(columnId))).values());
};
const uniqueCount = (columnId, leafRows) => {
  return new Set(leafRows.map((d) => d.getValue(columnId))).size;
};
const count = (_columnId, leafRows) => {
  return leafRows.length;
};
const aggregationFns = {
  sum,
  min,
  max,
  extent,
  mean,
  median,
  unique,
  uniqueCount,
  count
};
const ColumnGrouping = {
  getDefaultColumnDef: () => {
    return {
      aggregatedCell: (props) => {
        var _toString, _props$getValue;
        return (_toString = (_props$getValue = props.getValue()) == null || _props$getValue.toString == null ? void 0 : _props$getValue.toString()) != null ? _toString : null;
      },
      aggregationFn: "auto"
    };
  },
  getInitialState: (state) => {
    return {
      grouping: [],
      ...state
    };
  },
  getDefaultOptions: (table) => {
    return {
      onGroupingChange: makeStateUpdater("grouping", table),
      groupedColumnMode: "reorder"
    };
  },
  createColumn: (column, table) => {
    column.toggleGrouping = () => {
      table.setGrouping((old) => {
        if (old != null && old.includes(column.id)) {
          return old.filter((d) => d !== column.id);
        }
        return [...old != null ? old : [], column.id];
      });
    };
    column.getCanGroup = () => {
      var _column$columnDef$ena, _table$options$enable;
      return ((_column$columnDef$ena = column.columnDef.enableGrouping) != null ? _column$columnDef$ena : true) && ((_table$options$enable = table.options.enableGrouping) != null ? _table$options$enable : true) && (!!column.accessorFn || !!column.columnDef.getGroupingValue);
    };
    column.getIsGrouped = () => {
      var _table$getState$group;
      return (_table$getState$group = table.getState().grouping) == null ? void 0 : _table$getState$group.includes(column.id);
    };
    column.getGroupedIndex = () => {
      var _table$getState$group2;
      return (_table$getState$group2 = table.getState().grouping) == null ? void 0 : _table$getState$group2.indexOf(column.id);
    };
    column.getToggleGroupingHandler = () => {
      const canGroup = column.getCanGroup();
      return () => {
        if (!canGroup) return;
        column.toggleGrouping();
      };
    };
    column.getAutoAggregationFn = () => {
      const firstRow = table.getCoreRowModel().flatRows[0];
      const value = firstRow == null ? void 0 : firstRow.getValue(column.id);
      if (typeof value === "number") {
        return aggregationFns.sum;
      }
      if (Object.prototype.toString.call(value) === "[object Date]") {
        return aggregationFns.extent;
      }
    };
    column.getAggregationFn = () => {
      var _table$options$aggreg, _table$options$aggreg2;
      if (!column) {
        throw new Error();
      }
      return isFunction(column.columnDef.aggregationFn) ? column.columnDef.aggregationFn : column.columnDef.aggregationFn === "auto" ? column.getAutoAggregationFn() : (_table$options$aggreg = (_table$options$aggreg2 = table.options.aggregationFns) == null ? void 0 : _table$options$aggreg2[column.columnDef.aggregationFn]) != null ? _table$options$aggreg : aggregationFns[column.columnDef.aggregationFn];
    };
  },
  createTable: (table) => {
    table.setGrouping = (updater) => table.options.onGroupingChange == null ? void 0 : table.options.onGroupingChange(updater);
    table.resetGrouping = (defaultState) => {
      var _table$initialState$g, _table$initialState;
      table.setGrouping(defaultState ? [] : (_table$initialState$g = (_table$initialState = table.initialState) == null ? void 0 : _table$initialState.grouping) != null ? _table$initialState$g : []);
    };
    table.getPreGroupedRowModel = () => table.getFilteredRowModel();
    table.getGroupedRowModel = () => {
      if (!table._getGroupedRowModel && table.options.getGroupedRowModel) {
        table._getGroupedRowModel = table.options.getGroupedRowModel(table);
      }
      if (table.options.manualGrouping || !table._getGroupedRowModel) {
        return table.getPreGroupedRowModel();
      }
      return table._getGroupedRowModel();
    };
  },
  createRow: (row, table) => {
    row.getIsGrouped = () => !!row.groupingColumnId;
    row.getGroupingValue = (columnId) => {
      if (row._groupingValuesCache.hasOwnProperty(columnId)) {
        return row._groupingValuesCache[columnId];
      }
      const column = table.getColumn(columnId);
      if (!(column != null && column.columnDef.getGroupingValue)) {
        return row.getValue(columnId);
      }
      row._groupingValuesCache[columnId] = column.columnDef.getGroupingValue(row.original);
      return row._groupingValuesCache[columnId];
    };
    row._groupingValuesCache = {};
  },
  createCell: (cell, column, row, table) => {
    cell.getIsGrouped = () => column.getIsGrouped() && column.id === row.groupingColumnId;
    cell.getIsPlaceholder = () => !cell.getIsGrouped() && column.getIsGrouped();
    cell.getIsAggregated = () => {
      var _row$subRows;
      return !cell.getIsGrouped() && !cell.getIsPlaceholder() && !!((_row$subRows = row.subRows) != null && _row$subRows.length);
    };
  }
};
function orderColumns(leafColumns, grouping, groupedColumnMode) {
  if (!(grouping != null && grouping.length) || !groupedColumnMode) {
    return leafColumns;
  }
  const nonGroupingColumns = leafColumns.filter((col) => !grouping.includes(col.id));
  if (groupedColumnMode === "remove") {
    return nonGroupingColumns;
  }
  const groupingColumns = grouping.map((g) => leafColumns.find((col) => col.id === g)).filter(Boolean);
  return [...groupingColumns, ...nonGroupingColumns];
}
const ColumnOrdering = {
  getInitialState: (state) => {
    return {
      columnOrder: [],
      ...state
    };
  },
  getDefaultOptions: (table) => {
    return {
      onColumnOrderChange: makeStateUpdater("columnOrder", table)
    };
  },
  createColumn: (column, table) => {
    column.getIndex = memo$1((position) => [_getVisibleLeafColumns(table, position)], (columns) => columns.findIndex((d) => d.id === column.id), getMemoOptions(table.options, "debugColumns"));
    column.getIsFirstColumn = (position) => {
      var _columns$;
      const columns = _getVisibleLeafColumns(table, position);
      return ((_columns$ = columns[0]) == null ? void 0 : _columns$.id) === column.id;
    };
    column.getIsLastColumn = (position) => {
      var _columns;
      const columns = _getVisibleLeafColumns(table, position);
      return ((_columns = columns[columns.length - 1]) == null ? void 0 : _columns.id) === column.id;
    };
  },
  createTable: (table) => {
    table.setColumnOrder = (updater) => table.options.onColumnOrderChange == null ? void 0 : table.options.onColumnOrderChange(updater);
    table.resetColumnOrder = (defaultState) => {
      var _table$initialState$c;
      table.setColumnOrder(defaultState ? [] : (_table$initialState$c = table.initialState.columnOrder) != null ? _table$initialState$c : []);
    };
    table._getOrderColumnsFn = memo$1(() => [table.getState().columnOrder, table.getState().grouping, table.options.groupedColumnMode], (columnOrder, grouping, groupedColumnMode) => (columns) => {
      let orderedColumns = [];
      if (!(columnOrder != null && columnOrder.length)) {
        orderedColumns = columns;
      } else {
        const columnOrderCopy = [...columnOrder];
        const columnsCopy = [...columns];
        while (columnsCopy.length && columnOrderCopy.length) {
          const targetColumnId = columnOrderCopy.shift();
          const foundIndex = columnsCopy.findIndex((d) => d.id === targetColumnId);
          if (foundIndex > -1) {
            orderedColumns.push(columnsCopy.splice(foundIndex, 1)[0]);
          }
        }
        orderedColumns = [...orderedColumns, ...columnsCopy];
      }
      return orderColumns(orderedColumns, grouping, groupedColumnMode);
    }, getMemoOptions(table.options, "debugTable"));
  }
};
const getDefaultColumnPinningState = () => ({
  left: [],
  right: []
});
const ColumnPinning = {
  getInitialState: (state) => {
    return {
      columnPinning: getDefaultColumnPinningState(),
      ...state
    };
  },
  getDefaultOptions: (table) => {
    return {
      onColumnPinningChange: makeStateUpdater("columnPinning", table)
    };
  },
  createColumn: (column, table) => {
    column.pin = (position) => {
      const columnIds = column.getLeafColumns().map((d) => d.id).filter(Boolean);
      table.setColumnPinning((old) => {
        var _old$left3, _old$right3;
        if (position === "right") {
          var _old$left, _old$right;
          return {
            left: ((_old$left = old == null ? void 0 : old.left) != null ? _old$left : []).filter((d) => !(columnIds != null && columnIds.includes(d))),
            right: [...((_old$right = old == null ? void 0 : old.right) != null ? _old$right : []).filter((d) => !(columnIds != null && columnIds.includes(d))), ...columnIds]
          };
        }
        if (position === "left") {
          var _old$left2, _old$right2;
          return {
            left: [...((_old$left2 = old == null ? void 0 : old.left) != null ? _old$left2 : []).filter((d) => !(columnIds != null && columnIds.includes(d))), ...columnIds],
            right: ((_old$right2 = old == null ? void 0 : old.right) != null ? _old$right2 : []).filter((d) => !(columnIds != null && columnIds.includes(d)))
          };
        }
        return {
          left: ((_old$left3 = old == null ? void 0 : old.left) != null ? _old$left3 : []).filter((d) => !(columnIds != null && columnIds.includes(d))),
          right: ((_old$right3 = old == null ? void 0 : old.right) != null ? _old$right3 : []).filter((d) => !(columnIds != null && columnIds.includes(d)))
        };
      });
    };
    column.getCanPin = () => {
      const leafColumns = column.getLeafColumns();
      return leafColumns.some((d) => {
        var _d$columnDef$enablePi, _ref, _table$options$enable;
        return ((_d$columnDef$enablePi = d.columnDef.enablePinning) != null ? _d$columnDef$enablePi : true) && ((_ref = (_table$options$enable = table.options.enableColumnPinning) != null ? _table$options$enable : table.options.enablePinning) != null ? _ref : true);
      });
    };
    column.getIsPinned = () => {
      const leafColumnIds = column.getLeafColumns().map((d) => d.id);
      const {
        left,
        right
      } = table.getState().columnPinning;
      const isLeft = leafColumnIds.some((d) => left == null ? void 0 : left.includes(d));
      const isRight = leafColumnIds.some((d) => right == null ? void 0 : right.includes(d));
      return isLeft ? "left" : isRight ? "right" : false;
    };
    column.getPinnedIndex = () => {
      var _table$getState$colum, _table$getState$colum2;
      const position = column.getIsPinned();
      return position ? (_table$getState$colum = (_table$getState$colum2 = table.getState().columnPinning) == null || (_table$getState$colum2 = _table$getState$colum2[position]) == null ? void 0 : _table$getState$colum2.indexOf(column.id)) != null ? _table$getState$colum : -1 : 0;
    };
  },
  createRow: (row, table) => {
    row.getCenterVisibleCells = memo$1(() => [row._getAllVisibleCells(), table.getState().columnPinning.left, table.getState().columnPinning.right], (allCells, left, right) => {
      const leftAndRight = [...left != null ? left : [], ...right != null ? right : []];
      return allCells.filter((d) => !leftAndRight.includes(d.column.id));
    }, getMemoOptions(table.options, "debugRows"));
    row.getLeftVisibleCells = memo$1(() => [row._getAllVisibleCells(), table.getState().columnPinning.left], (allCells, left) => {
      const cells = (left != null ? left : []).map((columnId) => allCells.find((cell) => cell.column.id === columnId)).filter(Boolean).map((d) => ({
        ...d,
        position: "left"
      }));
      return cells;
    }, getMemoOptions(table.options, "debugRows"));
    row.getRightVisibleCells = memo$1(() => [row._getAllVisibleCells(), table.getState().columnPinning.right], (allCells, right) => {
      const cells = (right != null ? right : []).map((columnId) => allCells.find((cell) => cell.column.id === columnId)).filter(Boolean).map((d) => ({
        ...d,
        position: "right"
      }));
      return cells;
    }, getMemoOptions(table.options, "debugRows"));
  },
  createTable: (table) => {
    table.setColumnPinning = (updater) => table.options.onColumnPinningChange == null ? void 0 : table.options.onColumnPinningChange(updater);
    table.resetColumnPinning = (defaultState) => {
      var _table$initialState$c, _table$initialState;
      return table.setColumnPinning(defaultState ? getDefaultColumnPinningState() : (_table$initialState$c = (_table$initialState = table.initialState) == null ? void 0 : _table$initialState.columnPinning) != null ? _table$initialState$c : getDefaultColumnPinningState());
    };
    table.getIsSomeColumnsPinned = (position) => {
      var _pinningState$positio;
      const pinningState = table.getState().columnPinning;
      if (!position) {
        var _pinningState$left, _pinningState$right;
        return Boolean(((_pinningState$left = pinningState.left) == null ? void 0 : _pinningState$left.length) || ((_pinningState$right = pinningState.right) == null ? void 0 : _pinningState$right.length));
      }
      return Boolean((_pinningState$positio = pinningState[position]) == null ? void 0 : _pinningState$positio.length);
    };
    table.getLeftLeafColumns = memo$1(() => [table.getAllLeafColumns(), table.getState().columnPinning.left], (allColumns, left) => {
      return (left != null ? left : []).map((columnId) => allColumns.find((column) => column.id === columnId)).filter(Boolean);
    }, getMemoOptions(table.options, "debugColumns"));
    table.getRightLeafColumns = memo$1(() => [table.getAllLeafColumns(), table.getState().columnPinning.right], (allColumns, right) => {
      return (right != null ? right : []).map((columnId) => allColumns.find((column) => column.id === columnId)).filter(Boolean);
    }, getMemoOptions(table.options, "debugColumns"));
    table.getCenterLeafColumns = memo$1(() => [table.getAllLeafColumns(), table.getState().columnPinning.left, table.getState().columnPinning.right], (allColumns, left, right) => {
      const leftAndRight = [...left != null ? left : [], ...right != null ? right : []];
      return allColumns.filter((d) => !leftAndRight.includes(d.id));
    }, getMemoOptions(table.options, "debugColumns"));
  }
};
function safelyAccessDocument(_document) {
  return _document || (typeof document !== "undefined" ? document : null);
}
const defaultColumnSizing = {
  size: 150,
  minSize: 20,
  maxSize: Number.MAX_SAFE_INTEGER
};
const getDefaultColumnSizingInfoState = () => ({
  startOffset: null,
  startSize: null,
  deltaOffset: null,
  deltaPercentage: null,
  isResizingColumn: false,
  columnSizingStart: []
});
const ColumnSizing = {
  getDefaultColumnDef: () => {
    return defaultColumnSizing;
  },
  getInitialState: (state) => {
    return {
      columnSizing: {},
      columnSizingInfo: getDefaultColumnSizingInfoState(),
      ...state
    };
  },
  getDefaultOptions: (table) => {
    return {
      columnResizeMode: "onEnd",
      columnResizeDirection: "ltr",
      onColumnSizingChange: makeStateUpdater("columnSizing", table),
      onColumnSizingInfoChange: makeStateUpdater("columnSizingInfo", table)
    };
  },
  createColumn: (column, table) => {
    column.getSize = () => {
      var _column$columnDef$min, _ref, _column$columnDef$max;
      const columnSize = table.getState().columnSizing[column.id];
      return Math.min(Math.max((_column$columnDef$min = column.columnDef.minSize) != null ? _column$columnDef$min : defaultColumnSizing.minSize, (_ref = columnSize != null ? columnSize : column.columnDef.size) != null ? _ref : defaultColumnSizing.size), (_column$columnDef$max = column.columnDef.maxSize) != null ? _column$columnDef$max : defaultColumnSizing.maxSize);
    };
    column.getStart = memo$1((position) => [position, _getVisibleLeafColumns(table, position), table.getState().columnSizing], (position, columns) => columns.slice(0, column.getIndex(position)).reduce((sum2, column2) => sum2 + column2.getSize(), 0), getMemoOptions(table.options, "debugColumns"));
    column.getAfter = memo$1((position) => [position, _getVisibleLeafColumns(table, position), table.getState().columnSizing], (position, columns) => columns.slice(column.getIndex(position) + 1).reduce((sum2, column2) => sum2 + column2.getSize(), 0), getMemoOptions(table.options, "debugColumns"));
    column.resetSize = () => {
      table.setColumnSizing((_ref2) => {
        let {
          [column.id]: _,
          ...rest
        } = _ref2;
        return rest;
      });
    };
    column.getCanResize = () => {
      var _column$columnDef$ena, _table$options$enable;
      return ((_column$columnDef$ena = column.columnDef.enableResizing) != null ? _column$columnDef$ena : true) && ((_table$options$enable = table.options.enableColumnResizing) != null ? _table$options$enable : true);
    };
    column.getIsResizing = () => {
      return table.getState().columnSizingInfo.isResizingColumn === column.id;
    };
  },
  createHeader: (header, table) => {
    header.getSize = () => {
      let sum2 = 0;
      const recurse = (header2) => {
        if (header2.subHeaders.length) {
          header2.subHeaders.forEach(recurse);
        } else {
          var _header$column$getSiz;
          sum2 += (_header$column$getSiz = header2.column.getSize()) != null ? _header$column$getSiz : 0;
        }
      };
      recurse(header);
      return sum2;
    };
    header.getStart = () => {
      if (header.index > 0) {
        const prevSiblingHeader = header.headerGroup.headers[header.index - 1];
        return prevSiblingHeader.getStart() + prevSiblingHeader.getSize();
      }
      return 0;
    };
    header.getResizeHandler = (_contextDocument) => {
      const column = table.getColumn(header.column.id);
      const canResize = column == null ? void 0 : column.getCanResize();
      return (e) => {
        if (!column || !canResize) {
          return;
        }
        e.persist == null || e.persist();
        if (isTouchStartEvent(e)) {
          if (e.touches && e.touches.length > 1) {
            return;
          }
        }
        const startSize = header.getSize();
        const columnSizingStart = header ? header.getLeafHeaders().map((d) => [d.column.id, d.column.getSize()]) : [[column.id, column.getSize()]];
        const clientX = isTouchStartEvent(e) ? Math.round(e.touches[0].clientX) : e.clientX;
        const newColumnSizing = {};
        const updateOffset = (eventType, clientXPos) => {
          if (typeof clientXPos !== "number") {
            return;
          }
          table.setColumnSizingInfo((old) => {
            var _old$startOffset, _old$startSize;
            const deltaDirection = table.options.columnResizeDirection === "rtl" ? -1 : 1;
            const deltaOffset = (clientXPos - ((_old$startOffset = old == null ? void 0 : old.startOffset) != null ? _old$startOffset : 0)) * deltaDirection;
            const deltaPercentage = Math.max(deltaOffset / ((_old$startSize = old == null ? void 0 : old.startSize) != null ? _old$startSize : 0), -0.999999);
            old.columnSizingStart.forEach((_ref3) => {
              let [columnId, headerSize] = _ref3;
              newColumnSizing[columnId] = Math.round(Math.max(headerSize + headerSize * deltaPercentage, 0) * 100) / 100;
            });
            return {
              ...old,
              deltaOffset,
              deltaPercentage
            };
          });
          if (table.options.columnResizeMode === "onChange" || eventType === "end") {
            table.setColumnSizing((old) => ({
              ...old,
              ...newColumnSizing
            }));
          }
        };
        const onMove = (clientXPos) => updateOffset("move", clientXPos);
        const onEnd = (clientXPos) => {
          updateOffset("end", clientXPos);
          table.setColumnSizingInfo((old) => ({
            ...old,
            isResizingColumn: false,
            startOffset: null,
            startSize: null,
            deltaOffset: null,
            deltaPercentage: null,
            columnSizingStart: []
          }));
        };
        const contextDocument = safelyAccessDocument(_contextDocument);
        const mouseEvents = {
          moveHandler: (e2) => onMove(e2.clientX),
          upHandler: (e2) => {
            contextDocument == null || contextDocument.removeEventListener("mousemove", mouseEvents.moveHandler);
            contextDocument == null || contextDocument.removeEventListener("mouseup", mouseEvents.upHandler);
            onEnd(e2.clientX);
          }
        };
        const touchEvents = {
          moveHandler: (e2) => {
            if (e2.cancelable) {
              e2.preventDefault();
              e2.stopPropagation();
            }
            onMove(e2.touches[0].clientX);
            return false;
          },
          upHandler: (e2) => {
            var _e$touches$;
            contextDocument == null || contextDocument.removeEventListener("touchmove", touchEvents.moveHandler);
            contextDocument == null || contextDocument.removeEventListener("touchend", touchEvents.upHandler);
            if (e2.cancelable) {
              e2.preventDefault();
              e2.stopPropagation();
            }
            onEnd((_e$touches$ = e2.touches[0]) == null ? void 0 : _e$touches$.clientX);
          }
        };
        const passiveIfSupported = passiveEventSupported() ? {
          passive: false
        } : false;
        if (isTouchStartEvent(e)) {
          contextDocument == null || contextDocument.addEventListener("touchmove", touchEvents.moveHandler, passiveIfSupported);
          contextDocument == null || contextDocument.addEventListener("touchend", touchEvents.upHandler, passiveIfSupported);
        } else {
          contextDocument == null || contextDocument.addEventListener("mousemove", mouseEvents.moveHandler, passiveIfSupported);
          contextDocument == null || contextDocument.addEventListener("mouseup", mouseEvents.upHandler, passiveIfSupported);
        }
        table.setColumnSizingInfo((old) => ({
          ...old,
          startOffset: clientX,
          startSize,
          deltaOffset: 0,
          deltaPercentage: 0,
          columnSizingStart,
          isResizingColumn: column.id
        }));
      };
    };
  },
  createTable: (table) => {
    table.setColumnSizing = (updater) => table.options.onColumnSizingChange == null ? void 0 : table.options.onColumnSizingChange(updater);
    table.setColumnSizingInfo = (updater) => table.options.onColumnSizingInfoChange == null ? void 0 : table.options.onColumnSizingInfoChange(updater);
    table.resetColumnSizing = (defaultState) => {
      var _table$initialState$c;
      table.setColumnSizing(defaultState ? {} : (_table$initialState$c = table.initialState.columnSizing) != null ? _table$initialState$c : {});
    };
    table.resetHeaderSizeInfo = (defaultState) => {
      var _table$initialState$c2;
      table.setColumnSizingInfo(defaultState ? getDefaultColumnSizingInfoState() : (_table$initialState$c2 = table.initialState.columnSizingInfo) != null ? _table$initialState$c2 : getDefaultColumnSizingInfoState());
    };
    table.getTotalSize = () => {
      var _table$getHeaderGroup, _table$getHeaderGroup2;
      return (_table$getHeaderGroup = (_table$getHeaderGroup2 = table.getHeaderGroups()[0]) == null ? void 0 : _table$getHeaderGroup2.headers.reduce((sum2, header) => {
        return sum2 + header.getSize();
      }, 0)) != null ? _table$getHeaderGroup : 0;
    };
    table.getLeftTotalSize = () => {
      var _table$getLeftHeaderG, _table$getLeftHeaderG2;
      return (_table$getLeftHeaderG = (_table$getLeftHeaderG2 = table.getLeftHeaderGroups()[0]) == null ? void 0 : _table$getLeftHeaderG2.headers.reduce((sum2, header) => {
        return sum2 + header.getSize();
      }, 0)) != null ? _table$getLeftHeaderG : 0;
    };
    table.getCenterTotalSize = () => {
      var _table$getCenterHeade, _table$getCenterHeade2;
      return (_table$getCenterHeade = (_table$getCenterHeade2 = table.getCenterHeaderGroups()[0]) == null ? void 0 : _table$getCenterHeade2.headers.reduce((sum2, header) => {
        return sum2 + header.getSize();
      }, 0)) != null ? _table$getCenterHeade : 0;
    };
    table.getRightTotalSize = () => {
      var _table$getRightHeader, _table$getRightHeader2;
      return (_table$getRightHeader = (_table$getRightHeader2 = table.getRightHeaderGroups()[0]) == null ? void 0 : _table$getRightHeader2.headers.reduce((sum2, header) => {
        return sum2 + header.getSize();
      }, 0)) != null ? _table$getRightHeader : 0;
    };
  }
};
let passiveSupported = null;
function passiveEventSupported() {
  if (typeof passiveSupported === "boolean") return passiveSupported;
  let supported = false;
  try {
    const options = {
      get passive() {
        supported = true;
        return false;
      }
    };
    const noop2 = () => {
    };
    window.addEventListener("test", noop2, options);
    window.removeEventListener("test", noop2);
  } catch (err) {
    supported = false;
  }
  passiveSupported = supported;
  return passiveSupported;
}
function isTouchStartEvent(e) {
  return e.type === "touchstart";
}
const ColumnVisibility = {
  getInitialState: (state) => {
    return {
      columnVisibility: {},
      ...state
    };
  },
  getDefaultOptions: (table) => {
    return {
      onColumnVisibilityChange: makeStateUpdater("columnVisibility", table)
    };
  },
  createColumn: (column, table) => {
    column.toggleVisibility = (value) => {
      if (column.getCanHide()) {
        table.setColumnVisibility((old) => ({
          ...old,
          [column.id]: value != null ? value : !column.getIsVisible()
        }));
      }
    };
    column.getIsVisible = () => {
      var _ref, _table$getState$colum;
      const childColumns = column.columns;
      return (_ref = childColumns.length ? childColumns.some((c) => c.getIsVisible()) : (_table$getState$colum = table.getState().columnVisibility) == null ? void 0 : _table$getState$colum[column.id]) != null ? _ref : true;
    };
    column.getCanHide = () => {
      var _column$columnDef$ena, _table$options$enable;
      return ((_column$columnDef$ena = column.columnDef.enableHiding) != null ? _column$columnDef$ena : true) && ((_table$options$enable = table.options.enableHiding) != null ? _table$options$enable : true);
    };
    column.getToggleVisibilityHandler = () => {
      return (e) => {
        column.toggleVisibility == null || column.toggleVisibility(e.target.checked);
      };
    };
  },
  createRow: (row, table) => {
    row._getAllVisibleCells = memo$1(() => [row.getAllCells(), table.getState().columnVisibility], (cells) => {
      return cells.filter((cell) => cell.column.getIsVisible());
    }, getMemoOptions(table.options, "debugRows"));
    row.getVisibleCells = memo$1(() => [row.getLeftVisibleCells(), row.getCenterVisibleCells(), row.getRightVisibleCells()], (left, center, right) => [...left, ...center, ...right], getMemoOptions(table.options, "debugRows"));
  },
  createTable: (table) => {
    const makeVisibleColumnsMethod = (key, getColumns) => {
      return memo$1(() => [getColumns(), getColumns().filter((d) => d.getIsVisible()).map((d) => d.id).join("_")], (columns) => {
        return columns.filter((d) => d.getIsVisible == null ? void 0 : d.getIsVisible());
      }, getMemoOptions(table.options, "debugColumns"));
    };
    table.getVisibleFlatColumns = makeVisibleColumnsMethod("getVisibleFlatColumns", () => table.getAllFlatColumns());
    table.getVisibleLeafColumns = makeVisibleColumnsMethod("getVisibleLeafColumns", () => table.getAllLeafColumns());
    table.getLeftVisibleLeafColumns = makeVisibleColumnsMethod("getLeftVisibleLeafColumns", () => table.getLeftLeafColumns());
    table.getRightVisibleLeafColumns = makeVisibleColumnsMethod("getRightVisibleLeafColumns", () => table.getRightLeafColumns());
    table.getCenterVisibleLeafColumns = makeVisibleColumnsMethod("getCenterVisibleLeafColumns", () => table.getCenterLeafColumns());
    table.setColumnVisibility = (updater) => table.options.onColumnVisibilityChange == null ? void 0 : table.options.onColumnVisibilityChange(updater);
    table.resetColumnVisibility = (defaultState) => {
      var _table$initialState$c;
      table.setColumnVisibility(defaultState ? {} : (_table$initialState$c = table.initialState.columnVisibility) != null ? _table$initialState$c : {});
    };
    table.toggleAllColumnsVisible = (value) => {
      var _value;
      value = (_value = value) != null ? _value : !table.getIsAllColumnsVisible();
      table.setColumnVisibility(table.getAllLeafColumns().reduce((obj, column) => ({
        ...obj,
        [column.id]: !value ? !(column.getCanHide != null && column.getCanHide()) : value
      }), {}));
    };
    table.getIsAllColumnsVisible = () => !table.getAllLeafColumns().some((column) => !(column.getIsVisible != null && column.getIsVisible()));
    table.getIsSomeColumnsVisible = () => table.getAllLeafColumns().some((column) => column.getIsVisible == null ? void 0 : column.getIsVisible());
    table.getToggleAllColumnsVisibilityHandler = () => {
      return (e) => {
        var _target;
        table.toggleAllColumnsVisible((_target = e.target) == null ? void 0 : _target.checked);
      };
    };
  }
};
function _getVisibleLeafColumns(table, position) {
  return !position ? table.getVisibleLeafColumns() : position === "center" ? table.getCenterVisibleLeafColumns() : position === "left" ? table.getLeftVisibleLeafColumns() : table.getRightVisibleLeafColumns();
}
const GlobalFaceting = {
  createTable: (table) => {
    table._getGlobalFacetedRowModel = table.options.getFacetedRowModel && table.options.getFacetedRowModel(table, "__global__");
    table.getGlobalFacetedRowModel = () => {
      if (table.options.manualFiltering || !table._getGlobalFacetedRowModel) {
        return table.getPreFilteredRowModel();
      }
      return table._getGlobalFacetedRowModel();
    };
    table._getGlobalFacetedUniqueValues = table.options.getFacetedUniqueValues && table.options.getFacetedUniqueValues(table, "__global__");
    table.getGlobalFacetedUniqueValues = () => {
      if (!table._getGlobalFacetedUniqueValues) {
        return /* @__PURE__ */ new Map();
      }
      return table._getGlobalFacetedUniqueValues();
    };
    table._getGlobalFacetedMinMaxValues = table.options.getFacetedMinMaxValues && table.options.getFacetedMinMaxValues(table, "__global__");
    table.getGlobalFacetedMinMaxValues = () => {
      if (!table._getGlobalFacetedMinMaxValues) {
        return;
      }
      return table._getGlobalFacetedMinMaxValues();
    };
  }
};
const GlobalFiltering = {
  getInitialState: (state) => {
    return {
      globalFilter: void 0,
      ...state
    };
  },
  getDefaultOptions: (table) => {
    return {
      onGlobalFilterChange: makeStateUpdater("globalFilter", table),
      globalFilterFn: "auto",
      getColumnCanGlobalFilter: (column) => {
        var _table$getCoreRowMode;
        const value = (_table$getCoreRowMode = table.getCoreRowModel().flatRows[0]) == null || (_table$getCoreRowMode = _table$getCoreRowMode._getAllCellsByColumnId()[column.id]) == null ? void 0 : _table$getCoreRowMode.getValue();
        return typeof value === "string" || typeof value === "number";
      }
    };
  },
  createColumn: (column, table) => {
    column.getCanGlobalFilter = () => {
      var _column$columnDef$ena, _table$options$enable, _table$options$enable2, _table$options$getCol;
      return ((_column$columnDef$ena = column.columnDef.enableGlobalFilter) != null ? _column$columnDef$ena : true) && ((_table$options$enable = table.options.enableGlobalFilter) != null ? _table$options$enable : true) && ((_table$options$enable2 = table.options.enableFilters) != null ? _table$options$enable2 : true) && ((_table$options$getCol = table.options.getColumnCanGlobalFilter == null ? void 0 : table.options.getColumnCanGlobalFilter(column)) != null ? _table$options$getCol : true) && !!column.accessorFn;
    };
  },
  createTable: (table) => {
    table.getGlobalAutoFilterFn = () => {
      return filterFns.includesString;
    };
    table.getGlobalFilterFn = () => {
      var _table$options$filter, _table$options$filter2;
      const {
        globalFilterFn
      } = table.options;
      return isFunction(globalFilterFn) ? globalFilterFn : globalFilterFn === "auto" ? table.getGlobalAutoFilterFn() : (_table$options$filter = (_table$options$filter2 = table.options.filterFns) == null ? void 0 : _table$options$filter2[globalFilterFn]) != null ? _table$options$filter : filterFns[globalFilterFn];
    };
    table.setGlobalFilter = (updater) => {
      table.options.onGlobalFilterChange == null || table.options.onGlobalFilterChange(updater);
    };
    table.resetGlobalFilter = (defaultState) => {
      table.setGlobalFilter(defaultState ? void 0 : table.initialState.globalFilter);
    };
  }
};
const RowExpanding = {
  getInitialState: (state) => {
    return {
      expanded: {},
      ...state
    };
  },
  getDefaultOptions: (table) => {
    return {
      onExpandedChange: makeStateUpdater("expanded", table),
      paginateExpandedRows: true
    };
  },
  createTable: (table) => {
    let registered = false;
    let queued = false;
    table._autoResetExpanded = () => {
      var _ref, _table$options$autoRe;
      if (!registered) {
        table._queue(() => {
          registered = true;
        });
        return;
      }
      if ((_ref = (_table$options$autoRe = table.options.autoResetAll) != null ? _table$options$autoRe : table.options.autoResetExpanded) != null ? _ref : !table.options.manualExpanding) {
        if (queued) return;
        queued = true;
        table._queue(() => {
          table.resetExpanded();
          queued = false;
        });
      }
    };
    table.setExpanded = (updater) => table.options.onExpandedChange == null ? void 0 : table.options.onExpandedChange(updater);
    table.toggleAllRowsExpanded = (expanded) => {
      if (expanded != null ? expanded : !table.getIsAllRowsExpanded()) {
        table.setExpanded(true);
      } else {
        table.setExpanded({});
      }
    };
    table.resetExpanded = (defaultState) => {
      var _table$initialState$e, _table$initialState;
      table.setExpanded(defaultState ? {} : (_table$initialState$e = (_table$initialState = table.initialState) == null ? void 0 : _table$initialState.expanded) != null ? _table$initialState$e : {});
    };
    table.getCanSomeRowsExpand = () => {
      return table.getPrePaginationRowModel().flatRows.some((row) => row.getCanExpand());
    };
    table.getToggleAllRowsExpandedHandler = () => {
      return (e) => {
        e.persist == null || e.persist();
        table.toggleAllRowsExpanded();
      };
    };
    table.getIsSomeRowsExpanded = () => {
      const expanded = table.getState().expanded;
      return expanded === true || Object.values(expanded).some(Boolean);
    };
    table.getIsAllRowsExpanded = () => {
      const expanded = table.getState().expanded;
      if (typeof expanded === "boolean") {
        return expanded === true;
      }
      if (!Object.keys(expanded).length) {
        return false;
      }
      if (table.getRowModel().flatRows.some((row) => !row.getIsExpanded())) {
        return false;
      }
      return true;
    };
    table.getExpandedDepth = () => {
      let maxDepth = 0;
      const rowIds = table.getState().expanded === true ? Object.keys(table.getRowModel().rowsById) : Object.keys(table.getState().expanded);
      rowIds.forEach((id) => {
        const splitId = id.split(".");
        maxDepth = Math.max(maxDepth, splitId.length);
      });
      return maxDepth;
    };
    table.getPreExpandedRowModel = () => table.getSortedRowModel();
    table.getExpandedRowModel = () => {
      if (!table._getExpandedRowModel && table.options.getExpandedRowModel) {
        table._getExpandedRowModel = table.options.getExpandedRowModel(table);
      }
      if (table.options.manualExpanding || !table._getExpandedRowModel) {
        return table.getPreExpandedRowModel();
      }
      return table._getExpandedRowModel();
    };
  },
  createRow: (row, table) => {
    row.toggleExpanded = (expanded) => {
      table.setExpanded((old) => {
        var _expanded;
        const exists = old === true ? true : !!(old != null && old[row.id]);
        let oldExpanded = {};
        if (old === true) {
          Object.keys(table.getRowModel().rowsById).forEach((rowId) => {
            oldExpanded[rowId] = true;
          });
        } else {
          oldExpanded = old;
        }
        expanded = (_expanded = expanded) != null ? _expanded : !exists;
        if (!exists && expanded) {
          return {
            ...oldExpanded,
            [row.id]: true
          };
        }
        if (exists && !expanded) {
          const {
            [row.id]: _,
            ...rest
          } = oldExpanded;
          return rest;
        }
        return old;
      });
    };
    row.getIsExpanded = () => {
      var _table$options$getIsR;
      const expanded = table.getState().expanded;
      return !!((_table$options$getIsR = table.options.getIsRowExpanded == null ? void 0 : table.options.getIsRowExpanded(row)) != null ? _table$options$getIsR : expanded === true || (expanded == null ? void 0 : expanded[row.id]));
    };
    row.getCanExpand = () => {
      var _table$options$getRow, _table$options$enable, _row$subRows;
      return (_table$options$getRow = table.options.getRowCanExpand == null ? void 0 : table.options.getRowCanExpand(row)) != null ? _table$options$getRow : ((_table$options$enable = table.options.enableExpanding) != null ? _table$options$enable : true) && !!((_row$subRows = row.subRows) != null && _row$subRows.length);
    };
    row.getIsAllParentsExpanded = () => {
      let isFullyExpanded = true;
      let currentRow = row;
      while (isFullyExpanded && currentRow.parentId) {
        currentRow = table.getRow(currentRow.parentId, true);
        isFullyExpanded = currentRow.getIsExpanded();
      }
      return isFullyExpanded;
    };
    row.getToggleExpandedHandler = () => {
      const canExpand = row.getCanExpand();
      return () => {
        if (!canExpand) return;
        row.toggleExpanded();
      };
    };
  }
};
const defaultPageIndex = 0;
const defaultPageSize = 10;
const getDefaultPaginationState = () => ({
  pageIndex: defaultPageIndex,
  pageSize: defaultPageSize
});
const RowPagination = {
  getInitialState: (state) => {
    return {
      ...state,
      pagination: {
        ...getDefaultPaginationState(),
        ...state == null ? void 0 : state.pagination
      }
    };
  },
  getDefaultOptions: (table) => {
    return {
      onPaginationChange: makeStateUpdater("pagination", table)
    };
  },
  createTable: (table) => {
    let registered = false;
    let queued = false;
    table._autoResetPageIndex = () => {
      var _ref, _table$options$autoRe;
      if (!registered) {
        table._queue(() => {
          registered = true;
        });
        return;
      }
      if ((_ref = (_table$options$autoRe = table.options.autoResetAll) != null ? _table$options$autoRe : table.options.autoResetPageIndex) != null ? _ref : !table.options.manualPagination) {
        if (queued) return;
        queued = true;
        table._queue(() => {
          table.resetPageIndex();
          queued = false;
        });
      }
    };
    table.setPagination = (updater) => {
      const safeUpdater = (old) => {
        let newState = functionalUpdate(updater, old);
        return newState;
      };
      return table.options.onPaginationChange == null ? void 0 : table.options.onPaginationChange(safeUpdater);
    };
    table.resetPagination = (defaultState) => {
      var _table$initialState$p;
      table.setPagination(defaultState ? getDefaultPaginationState() : (_table$initialState$p = table.initialState.pagination) != null ? _table$initialState$p : getDefaultPaginationState());
    };
    table.setPageIndex = (updater) => {
      table.setPagination((old) => {
        let pageIndex = functionalUpdate(updater, old.pageIndex);
        const maxPageIndex = typeof table.options.pageCount === "undefined" || table.options.pageCount === -1 ? Number.MAX_SAFE_INTEGER : table.options.pageCount - 1;
        pageIndex = Math.max(0, Math.min(pageIndex, maxPageIndex));
        return {
          ...old,
          pageIndex
        };
      });
    };
    table.resetPageIndex = (defaultState) => {
      var _table$initialState$p2, _table$initialState;
      table.setPageIndex(defaultState ? defaultPageIndex : (_table$initialState$p2 = (_table$initialState = table.initialState) == null || (_table$initialState = _table$initialState.pagination) == null ? void 0 : _table$initialState.pageIndex) != null ? _table$initialState$p2 : defaultPageIndex);
    };
    table.resetPageSize = (defaultState) => {
      var _table$initialState$p3, _table$initialState2;
      table.setPageSize(defaultState ? defaultPageSize : (_table$initialState$p3 = (_table$initialState2 = table.initialState) == null || (_table$initialState2 = _table$initialState2.pagination) == null ? void 0 : _table$initialState2.pageSize) != null ? _table$initialState$p3 : defaultPageSize);
    };
    table.setPageSize = (updater) => {
      table.setPagination((old) => {
        const pageSize = Math.max(1, functionalUpdate(updater, old.pageSize));
        const topRowIndex = old.pageSize * old.pageIndex;
        const pageIndex = Math.floor(topRowIndex / pageSize);
        return {
          ...old,
          pageIndex,
          pageSize
        };
      });
    };
    table.setPageCount = (updater) => table.setPagination((old) => {
      var _table$options$pageCo;
      let newPageCount = functionalUpdate(updater, (_table$options$pageCo = table.options.pageCount) != null ? _table$options$pageCo : -1);
      if (typeof newPageCount === "number") {
        newPageCount = Math.max(-1, newPageCount);
      }
      return {
        ...old,
        pageCount: newPageCount
      };
    });
    table.getPageOptions = memo$1(() => [table.getPageCount()], (pageCount) => {
      let pageOptions = [];
      if (pageCount && pageCount > 0) {
        pageOptions = [...new Array(pageCount)].fill(null).map((_, i) => i);
      }
      return pageOptions;
    }, getMemoOptions(table.options, "debugTable"));
    table.getCanPreviousPage = () => table.getState().pagination.pageIndex > 0;
    table.getCanNextPage = () => {
      const {
        pageIndex
      } = table.getState().pagination;
      const pageCount = table.getPageCount();
      if (pageCount === -1) {
        return true;
      }
      if (pageCount === 0) {
        return false;
      }
      return pageIndex < pageCount - 1;
    };
    table.previousPage = () => {
      return table.setPageIndex((old) => old - 1);
    };
    table.nextPage = () => {
      return table.setPageIndex((old) => {
        return old + 1;
      });
    };
    table.firstPage = () => {
      return table.setPageIndex(0);
    };
    table.lastPage = () => {
      return table.setPageIndex(table.getPageCount() - 1);
    };
    table.getPrePaginationRowModel = () => table.getExpandedRowModel();
    table.getPaginationRowModel = () => {
      if (!table._getPaginationRowModel && table.options.getPaginationRowModel) {
        table._getPaginationRowModel = table.options.getPaginationRowModel(table);
      }
      if (table.options.manualPagination || !table._getPaginationRowModel) {
        return table.getPrePaginationRowModel();
      }
      return table._getPaginationRowModel();
    };
    table.getPageCount = () => {
      var _table$options$pageCo2;
      return (_table$options$pageCo2 = table.options.pageCount) != null ? _table$options$pageCo2 : Math.ceil(table.getRowCount() / table.getState().pagination.pageSize);
    };
    table.getRowCount = () => {
      var _table$options$rowCou;
      return (_table$options$rowCou = table.options.rowCount) != null ? _table$options$rowCou : table.getPrePaginationRowModel().rows.length;
    };
  }
};
const getDefaultRowPinningState = () => ({
  top: [],
  bottom: []
});
const RowPinning = {
  getInitialState: (state) => {
    return {
      rowPinning: getDefaultRowPinningState(),
      ...state
    };
  },
  getDefaultOptions: (table) => {
    return {
      onRowPinningChange: makeStateUpdater("rowPinning", table)
    };
  },
  createRow: (row, table) => {
    row.pin = (position, includeLeafRows, includeParentRows) => {
      const leafRowIds = includeLeafRows ? row.getLeafRows().map((_ref) => {
        let {
          id
        } = _ref;
        return id;
      }) : [];
      const parentRowIds = includeParentRows ? row.getParentRows().map((_ref2) => {
        let {
          id
        } = _ref2;
        return id;
      }) : [];
      const rowIds = /* @__PURE__ */ new Set([...parentRowIds, row.id, ...leafRowIds]);
      table.setRowPinning((old) => {
        var _old$top3, _old$bottom3;
        if (position === "bottom") {
          var _old$top, _old$bottom;
          return {
            top: ((_old$top = old == null ? void 0 : old.top) != null ? _old$top : []).filter((d) => !(rowIds != null && rowIds.has(d))),
            bottom: [...((_old$bottom = old == null ? void 0 : old.bottom) != null ? _old$bottom : []).filter((d) => !(rowIds != null && rowIds.has(d))), ...Array.from(rowIds)]
          };
        }
        if (position === "top") {
          var _old$top2, _old$bottom2;
          return {
            top: [...((_old$top2 = old == null ? void 0 : old.top) != null ? _old$top2 : []).filter((d) => !(rowIds != null && rowIds.has(d))), ...Array.from(rowIds)],
            bottom: ((_old$bottom2 = old == null ? void 0 : old.bottom) != null ? _old$bottom2 : []).filter((d) => !(rowIds != null && rowIds.has(d)))
          };
        }
        return {
          top: ((_old$top3 = old == null ? void 0 : old.top) != null ? _old$top3 : []).filter((d) => !(rowIds != null && rowIds.has(d))),
          bottom: ((_old$bottom3 = old == null ? void 0 : old.bottom) != null ? _old$bottom3 : []).filter((d) => !(rowIds != null && rowIds.has(d)))
        };
      });
    };
    row.getCanPin = () => {
      var _ref3;
      const {
        enableRowPinning,
        enablePinning
      } = table.options;
      if (typeof enableRowPinning === "function") {
        return enableRowPinning(row);
      }
      return (_ref3 = enableRowPinning != null ? enableRowPinning : enablePinning) != null ? _ref3 : true;
    };
    row.getIsPinned = () => {
      const rowIds = [row.id];
      const {
        top,
        bottom
      } = table.getState().rowPinning;
      const isTop = rowIds.some((d) => top == null ? void 0 : top.includes(d));
      const isBottom = rowIds.some((d) => bottom == null ? void 0 : bottom.includes(d));
      return isTop ? "top" : isBottom ? "bottom" : false;
    };
    row.getPinnedIndex = () => {
      var _ref4, _visiblePinnedRowIds$;
      const position = row.getIsPinned();
      if (!position) return -1;
      const visiblePinnedRowIds = (_ref4 = position === "top" ? table.getTopRows() : table.getBottomRows()) == null ? void 0 : _ref4.map((_ref5) => {
        let {
          id
        } = _ref5;
        return id;
      });
      return (_visiblePinnedRowIds$ = visiblePinnedRowIds == null ? void 0 : visiblePinnedRowIds.indexOf(row.id)) != null ? _visiblePinnedRowIds$ : -1;
    };
  },
  createTable: (table) => {
    table.setRowPinning = (updater) => table.options.onRowPinningChange == null ? void 0 : table.options.onRowPinningChange(updater);
    table.resetRowPinning = (defaultState) => {
      var _table$initialState$r, _table$initialState;
      return table.setRowPinning(defaultState ? getDefaultRowPinningState() : (_table$initialState$r = (_table$initialState = table.initialState) == null ? void 0 : _table$initialState.rowPinning) != null ? _table$initialState$r : getDefaultRowPinningState());
    };
    table.getIsSomeRowsPinned = (position) => {
      var _pinningState$positio;
      const pinningState = table.getState().rowPinning;
      if (!position) {
        var _pinningState$top, _pinningState$bottom;
        return Boolean(((_pinningState$top = pinningState.top) == null ? void 0 : _pinningState$top.length) || ((_pinningState$bottom = pinningState.bottom) == null ? void 0 : _pinningState$bottom.length));
      }
      return Boolean((_pinningState$positio = pinningState[position]) == null ? void 0 : _pinningState$positio.length);
    };
    table._getPinnedRows = (visibleRows, pinnedRowIds, position) => {
      var _table$options$keepPi;
      const rows = ((_table$options$keepPi = table.options.keepPinnedRows) != null ? _table$options$keepPi : true) ? (
        //get all rows that are pinned even if they would not be otherwise visible
        //account for expanded parent rows, but not pagination or filtering
        (pinnedRowIds != null ? pinnedRowIds : []).map((rowId) => {
          const row = table.getRow(rowId, true);
          return row.getIsAllParentsExpanded() ? row : null;
        })
      ) : (
        //else get only visible rows that are pinned
        (pinnedRowIds != null ? pinnedRowIds : []).map((rowId) => visibleRows.find((row) => row.id === rowId))
      );
      return rows.filter(Boolean).map((d) => ({
        ...d,
        position
      }));
    };
    table.getTopRows = memo$1(() => [table.getRowModel().rows, table.getState().rowPinning.top], (allRows, topPinnedRowIds) => table._getPinnedRows(allRows, topPinnedRowIds, "top"), getMemoOptions(table.options, "debugRows"));
    table.getBottomRows = memo$1(() => [table.getRowModel().rows, table.getState().rowPinning.bottom], (allRows, bottomPinnedRowIds) => table._getPinnedRows(allRows, bottomPinnedRowIds, "bottom"), getMemoOptions(table.options, "debugRows"));
    table.getCenterRows = memo$1(() => [table.getRowModel().rows, table.getState().rowPinning.top, table.getState().rowPinning.bottom], (allRows, top, bottom) => {
      const topAndBottom = /* @__PURE__ */ new Set([...top != null ? top : [], ...bottom != null ? bottom : []]);
      return allRows.filter((d) => !topAndBottom.has(d.id));
    }, getMemoOptions(table.options, "debugRows"));
  }
};
const RowSelection = {
  getInitialState: (state) => {
    return {
      rowSelection: {},
      ...state
    };
  },
  getDefaultOptions: (table) => {
    return {
      onRowSelectionChange: makeStateUpdater("rowSelection", table),
      enableRowSelection: true,
      enableMultiRowSelection: true,
      enableSubRowSelection: true
      // enableGroupingRowSelection: false,
      // isAdditiveSelectEvent: (e: unknown) => !!e.metaKey,
      // isInclusiveSelectEvent: (e: unknown) => !!e.shiftKey,
    };
  },
  createTable: (table) => {
    table.setRowSelection = (updater) => table.options.onRowSelectionChange == null ? void 0 : table.options.onRowSelectionChange(updater);
    table.resetRowSelection = (defaultState) => {
      var _table$initialState$r;
      return table.setRowSelection(defaultState ? {} : (_table$initialState$r = table.initialState.rowSelection) != null ? _table$initialState$r : {});
    };
    table.toggleAllRowsSelected = (value) => {
      table.setRowSelection((old) => {
        value = typeof value !== "undefined" ? value : !table.getIsAllRowsSelected();
        const rowSelection = {
          ...old
        };
        const preGroupedFlatRows = table.getPreGroupedRowModel().flatRows;
        if (value) {
          preGroupedFlatRows.forEach((row) => {
            if (!row.getCanSelect()) {
              return;
            }
            rowSelection[row.id] = true;
          });
        } else {
          preGroupedFlatRows.forEach((row) => {
            delete rowSelection[row.id];
          });
        }
        return rowSelection;
      });
    };
    table.toggleAllPageRowsSelected = (value) => table.setRowSelection((old) => {
      const resolvedValue = typeof value !== "undefined" ? value : !table.getIsAllPageRowsSelected();
      const rowSelection = {
        ...old
      };
      table.getRowModel().rows.forEach((row) => {
        mutateRowIsSelected(rowSelection, row.id, resolvedValue, true, table);
      });
      return rowSelection;
    });
    table.getPreSelectedRowModel = () => table.getCoreRowModel();
    table.getSelectedRowModel = memo$1(() => [table.getState().rowSelection, table.getCoreRowModel()], (rowSelection, rowModel) => {
      if (!Object.keys(rowSelection).length) {
        return {
          rows: [],
          flatRows: [],
          rowsById: {}
        };
      }
      return selectRowsFn(table, rowModel);
    }, getMemoOptions(table.options, "debugTable"));
    table.getFilteredSelectedRowModel = memo$1(() => [table.getState().rowSelection, table.getFilteredRowModel()], (rowSelection, rowModel) => {
      if (!Object.keys(rowSelection).length) {
        return {
          rows: [],
          flatRows: [],
          rowsById: {}
        };
      }
      return selectRowsFn(table, rowModel);
    }, getMemoOptions(table.options, "debugTable"));
    table.getGroupedSelectedRowModel = memo$1(() => [table.getState().rowSelection, table.getSortedRowModel()], (rowSelection, rowModel) => {
      if (!Object.keys(rowSelection).length) {
        return {
          rows: [],
          flatRows: [],
          rowsById: {}
        };
      }
      return selectRowsFn(table, rowModel);
    }, getMemoOptions(table.options, "debugTable"));
    table.getIsAllRowsSelected = () => {
      const preGroupedFlatRows = table.getFilteredRowModel().flatRows;
      const {
        rowSelection
      } = table.getState();
      let isAllRowsSelected = Boolean(preGroupedFlatRows.length && Object.keys(rowSelection).length);
      if (isAllRowsSelected) {
        if (preGroupedFlatRows.some((row) => row.getCanSelect() && !rowSelection[row.id])) {
          isAllRowsSelected = false;
        }
      }
      return isAllRowsSelected;
    };
    table.getIsAllPageRowsSelected = () => {
      const paginationFlatRows = table.getPaginationRowModel().flatRows.filter((row) => row.getCanSelect());
      const {
        rowSelection
      } = table.getState();
      let isAllPageRowsSelected = !!paginationFlatRows.length;
      if (isAllPageRowsSelected && paginationFlatRows.some((row) => !rowSelection[row.id])) {
        isAllPageRowsSelected = false;
      }
      return isAllPageRowsSelected;
    };
    table.getIsSomeRowsSelected = () => {
      var _table$getState$rowSe;
      const totalSelected = Object.keys((_table$getState$rowSe = table.getState().rowSelection) != null ? _table$getState$rowSe : {}).length;
      return totalSelected > 0 && totalSelected < table.getFilteredRowModel().flatRows.length;
    };
    table.getIsSomePageRowsSelected = () => {
      const paginationFlatRows = table.getPaginationRowModel().flatRows;
      return table.getIsAllPageRowsSelected() ? false : paginationFlatRows.filter((row) => row.getCanSelect()).some((d) => d.getIsSelected() || d.getIsSomeSelected());
    };
    table.getToggleAllRowsSelectedHandler = () => {
      return (e) => {
        table.toggleAllRowsSelected(e.target.checked);
      };
    };
    table.getToggleAllPageRowsSelectedHandler = () => {
      return (e) => {
        table.toggleAllPageRowsSelected(e.target.checked);
      };
    };
  },
  createRow: (row, table) => {
    row.toggleSelected = (value, opts) => {
      const isSelected = row.getIsSelected();
      table.setRowSelection((old) => {
        var _opts$selectChildren;
        value = typeof value !== "undefined" ? value : !isSelected;
        if (row.getCanSelect() && isSelected === value) {
          return old;
        }
        const selectedRowIds = {
          ...old
        };
        mutateRowIsSelected(selectedRowIds, row.id, value, (_opts$selectChildren = opts == null ? void 0 : opts.selectChildren) != null ? _opts$selectChildren : true, table);
        return selectedRowIds;
      });
    };
    row.getIsSelected = () => {
      const {
        rowSelection
      } = table.getState();
      return isRowSelected(row, rowSelection);
    };
    row.getIsSomeSelected = () => {
      const {
        rowSelection
      } = table.getState();
      return isSubRowSelected(row, rowSelection) === "some";
    };
    row.getIsAllSubRowsSelected = () => {
      const {
        rowSelection
      } = table.getState();
      return isSubRowSelected(row, rowSelection) === "all";
    };
    row.getCanSelect = () => {
      var _table$options$enable;
      if (typeof table.options.enableRowSelection === "function") {
        return table.options.enableRowSelection(row);
      }
      return (_table$options$enable = table.options.enableRowSelection) != null ? _table$options$enable : true;
    };
    row.getCanSelectSubRows = () => {
      var _table$options$enable2;
      if (typeof table.options.enableSubRowSelection === "function") {
        return table.options.enableSubRowSelection(row);
      }
      return (_table$options$enable2 = table.options.enableSubRowSelection) != null ? _table$options$enable2 : true;
    };
    row.getCanMultiSelect = () => {
      var _table$options$enable3;
      if (typeof table.options.enableMultiRowSelection === "function") {
        return table.options.enableMultiRowSelection(row);
      }
      return (_table$options$enable3 = table.options.enableMultiRowSelection) != null ? _table$options$enable3 : true;
    };
    row.getToggleSelectedHandler = () => {
      const canSelect = row.getCanSelect();
      return (e) => {
        var _target;
        if (!canSelect) return;
        row.toggleSelected((_target = e.target) == null ? void 0 : _target.checked);
      };
    };
  }
};
const mutateRowIsSelected = (selectedRowIds, id, value, includeChildren, table) => {
  var _row$subRows;
  const row = table.getRow(id, true);
  if (value) {
    if (!row.getCanMultiSelect()) {
      Object.keys(selectedRowIds).forEach((key) => delete selectedRowIds[key]);
    }
    if (row.getCanSelect()) {
      selectedRowIds[id] = true;
    }
  } else {
    delete selectedRowIds[id];
  }
  if (includeChildren && (_row$subRows = row.subRows) != null && _row$subRows.length && row.getCanSelectSubRows()) {
    row.subRows.forEach((row2) => mutateRowIsSelected(selectedRowIds, row2.id, value, includeChildren, table));
  }
};
function selectRowsFn(table, rowModel) {
  const rowSelection = table.getState().rowSelection;
  const newSelectedFlatRows = [];
  const newSelectedRowsById = {};
  const recurseRows = function(rows, depth) {
    return rows.map((row) => {
      var _row$subRows2;
      const isSelected = isRowSelected(row, rowSelection);
      if (isSelected) {
        newSelectedFlatRows.push(row);
        newSelectedRowsById[row.id] = row;
      }
      if ((_row$subRows2 = row.subRows) != null && _row$subRows2.length) {
        row = {
          ...row,
          subRows: recurseRows(row.subRows)
        };
      }
      if (isSelected) {
        return row;
      }
    }).filter(Boolean);
  };
  return {
    rows: recurseRows(rowModel.rows),
    flatRows: newSelectedFlatRows,
    rowsById: newSelectedRowsById
  };
}
function isRowSelected(row, selection) {
  var _selection$row$id;
  return (_selection$row$id = selection[row.id]) != null ? _selection$row$id : false;
}
function isSubRowSelected(row, selection, table) {
  var _row$subRows3;
  if (!((_row$subRows3 = row.subRows) != null && _row$subRows3.length)) return false;
  let allChildrenSelected = true;
  let someSelected = false;
  row.subRows.forEach((subRow) => {
    if (someSelected && !allChildrenSelected) {
      return;
    }
    if (subRow.getCanSelect()) {
      if (isRowSelected(subRow, selection)) {
        someSelected = true;
      } else {
        allChildrenSelected = false;
      }
    }
    if (subRow.subRows && subRow.subRows.length) {
      const subRowChildrenSelected = isSubRowSelected(subRow, selection);
      if (subRowChildrenSelected === "all") {
        someSelected = true;
      } else if (subRowChildrenSelected === "some") {
        someSelected = true;
        allChildrenSelected = false;
      } else {
        allChildrenSelected = false;
      }
    }
  });
  return allChildrenSelected ? "all" : someSelected ? "some" : false;
}
const reSplitAlphaNumeric = /([0-9]+)/gm;
const alphanumeric = (rowA, rowB, columnId) => {
  return compareAlphanumeric(toString(rowA.getValue(columnId)).toLowerCase(), toString(rowB.getValue(columnId)).toLowerCase());
};
const alphanumericCaseSensitive = (rowA, rowB, columnId) => {
  return compareAlphanumeric(toString(rowA.getValue(columnId)), toString(rowB.getValue(columnId)));
};
const text = (rowA, rowB, columnId) => {
  return compareBasic(toString(rowA.getValue(columnId)).toLowerCase(), toString(rowB.getValue(columnId)).toLowerCase());
};
const textCaseSensitive = (rowA, rowB, columnId) => {
  return compareBasic(toString(rowA.getValue(columnId)), toString(rowB.getValue(columnId)));
};
const datetime = (rowA, rowB, columnId) => {
  const a = rowA.getValue(columnId);
  const b = rowB.getValue(columnId);
  return a > b ? 1 : a < b ? -1 : 0;
};
const basic = (rowA, rowB, columnId) => {
  return compareBasic(rowA.getValue(columnId), rowB.getValue(columnId));
};
function compareBasic(a, b) {
  return a === b ? 0 : a > b ? 1 : -1;
}
function toString(a) {
  if (typeof a === "number") {
    if (isNaN(a) || a === Infinity || a === -Infinity) {
      return "";
    }
    return String(a);
  }
  if (typeof a === "string") {
    return a;
  }
  return "";
}
function compareAlphanumeric(aStr, bStr) {
  const a = aStr.split(reSplitAlphaNumeric).filter(Boolean);
  const b = bStr.split(reSplitAlphaNumeric).filter(Boolean);
  while (a.length && b.length) {
    const aa = a.shift();
    const bb = b.shift();
    const an = parseInt(aa, 10);
    const bn = parseInt(bb, 10);
    const combo = [an, bn].sort();
    if (isNaN(combo[0])) {
      if (aa > bb) {
        return 1;
      }
      if (bb > aa) {
        return -1;
      }
      continue;
    }
    if (isNaN(combo[1])) {
      return isNaN(an) ? -1 : 1;
    }
    if (an > bn) {
      return 1;
    }
    if (bn > an) {
      return -1;
    }
  }
  return a.length - b.length;
}
const sortingFns = {
  alphanumeric,
  alphanumericCaseSensitive,
  text,
  textCaseSensitive,
  datetime,
  basic
};
const RowSorting = {
  getInitialState: (state) => {
    return {
      sorting: [],
      ...state
    };
  },
  getDefaultColumnDef: () => {
    return {
      sortingFn: "auto",
      sortUndefined: 1
    };
  },
  getDefaultOptions: (table) => {
    return {
      onSortingChange: makeStateUpdater("sorting", table),
      isMultiSortEvent: (e) => {
        return e.shiftKey;
      }
    };
  },
  createColumn: (column, table) => {
    column.getAutoSortingFn = () => {
      const firstRows = table.getFilteredRowModel().flatRows.slice(10);
      let isString = false;
      for (const row of firstRows) {
        const value = row == null ? void 0 : row.getValue(column.id);
        if (Object.prototype.toString.call(value) === "[object Date]") {
          return sortingFns.datetime;
        }
        if (typeof value === "string") {
          isString = true;
          if (value.split(reSplitAlphaNumeric).length > 1) {
            return sortingFns.alphanumeric;
          }
        }
      }
      if (isString) {
        return sortingFns.text;
      }
      return sortingFns.basic;
    };
    column.getAutoSortDir = () => {
      const firstRow = table.getFilteredRowModel().flatRows[0];
      const value = firstRow == null ? void 0 : firstRow.getValue(column.id);
      if (typeof value === "string") {
        return "asc";
      }
      return "desc";
    };
    column.getSortingFn = () => {
      var _table$options$sortin, _table$options$sortin2;
      if (!column) {
        throw new Error();
      }
      return isFunction(column.columnDef.sortingFn) ? column.columnDef.sortingFn : column.columnDef.sortingFn === "auto" ? column.getAutoSortingFn() : (_table$options$sortin = (_table$options$sortin2 = table.options.sortingFns) == null ? void 0 : _table$options$sortin2[column.columnDef.sortingFn]) != null ? _table$options$sortin : sortingFns[column.columnDef.sortingFn];
    };
    column.toggleSorting = (desc, multi) => {
      const nextSortingOrder = column.getNextSortingOrder();
      const hasManualValue = typeof desc !== "undefined" && desc !== null;
      table.setSorting((old) => {
        const existingSorting = old == null ? void 0 : old.find((d) => d.id === column.id);
        const existingIndex = old == null ? void 0 : old.findIndex((d) => d.id === column.id);
        let newSorting = [];
        let sortAction;
        let nextDesc = hasManualValue ? desc : nextSortingOrder === "desc";
        if (old != null && old.length && column.getCanMultiSort() && multi) {
          if (existingSorting) {
            sortAction = "toggle";
          } else {
            sortAction = "add";
          }
        } else {
          if (old != null && old.length && existingIndex !== old.length - 1) {
            sortAction = "replace";
          } else if (existingSorting) {
            sortAction = "toggle";
          } else {
            sortAction = "replace";
          }
        }
        if (sortAction === "toggle") {
          if (!hasManualValue) {
            if (!nextSortingOrder) {
              sortAction = "remove";
            }
          }
        }
        if (sortAction === "add") {
          var _table$options$maxMul;
          newSorting = [...old, {
            id: column.id,
            desc: nextDesc
          }];
          newSorting.splice(0, newSorting.length - ((_table$options$maxMul = table.options.maxMultiSortColCount) != null ? _table$options$maxMul : Number.MAX_SAFE_INTEGER));
        } else if (sortAction === "toggle") {
          newSorting = old.map((d) => {
            if (d.id === column.id) {
              return {
                ...d,
                desc: nextDesc
              };
            }
            return d;
          });
        } else if (sortAction === "remove") {
          newSorting = old.filter((d) => d.id !== column.id);
        } else {
          newSorting = [{
            id: column.id,
            desc: nextDesc
          }];
        }
        return newSorting;
      });
    };
    column.getFirstSortDir = () => {
      var _ref, _column$columnDef$sor;
      const sortDescFirst = (_ref = (_column$columnDef$sor = column.columnDef.sortDescFirst) != null ? _column$columnDef$sor : table.options.sortDescFirst) != null ? _ref : column.getAutoSortDir() === "desc";
      return sortDescFirst ? "desc" : "asc";
    };
    column.getNextSortingOrder = (multi) => {
      var _table$options$enable, _table$options$enable2;
      const firstSortDirection = column.getFirstSortDir();
      const isSorted = column.getIsSorted();
      if (!isSorted) {
        return firstSortDirection;
      }
      if (isSorted !== firstSortDirection && ((_table$options$enable = table.options.enableSortingRemoval) != null ? _table$options$enable : true) && // If enableSortRemove, enable in general
      (multi ? (_table$options$enable2 = table.options.enableMultiRemove) != null ? _table$options$enable2 : true : true)) {
        return false;
      }
      return isSorted === "desc" ? "asc" : "desc";
    };
    column.getCanSort = () => {
      var _column$columnDef$ena, _table$options$enable3;
      return ((_column$columnDef$ena = column.columnDef.enableSorting) != null ? _column$columnDef$ena : true) && ((_table$options$enable3 = table.options.enableSorting) != null ? _table$options$enable3 : true) && !!column.accessorFn;
    };
    column.getCanMultiSort = () => {
      var _ref2, _column$columnDef$ena2;
      return (_ref2 = (_column$columnDef$ena2 = column.columnDef.enableMultiSort) != null ? _column$columnDef$ena2 : table.options.enableMultiSort) != null ? _ref2 : !!column.accessorFn;
    };
    column.getIsSorted = () => {
      var _table$getState$sorti;
      const columnSort = (_table$getState$sorti = table.getState().sorting) == null ? void 0 : _table$getState$sorti.find((d) => d.id === column.id);
      return !columnSort ? false : columnSort.desc ? "desc" : "asc";
    };
    column.getSortIndex = () => {
      var _table$getState$sorti2, _table$getState$sorti3;
      return (_table$getState$sorti2 = (_table$getState$sorti3 = table.getState().sorting) == null ? void 0 : _table$getState$sorti3.findIndex((d) => d.id === column.id)) != null ? _table$getState$sorti2 : -1;
    };
    column.clearSorting = () => {
      table.setSorting((old) => old != null && old.length ? old.filter((d) => d.id !== column.id) : []);
    };
    column.getToggleSortingHandler = () => {
      const canSort = column.getCanSort();
      return (e) => {
        if (!canSort) return;
        e.persist == null || e.persist();
        column.toggleSorting == null || column.toggleSorting(void 0, column.getCanMultiSort() ? table.options.isMultiSortEvent == null ? void 0 : table.options.isMultiSortEvent(e) : false);
      };
    };
  },
  createTable: (table) => {
    table.setSorting = (updater) => table.options.onSortingChange == null ? void 0 : table.options.onSortingChange(updater);
    table.resetSorting = (defaultState) => {
      var _table$initialState$s, _table$initialState;
      table.setSorting(defaultState ? [] : (_table$initialState$s = (_table$initialState = table.initialState) == null ? void 0 : _table$initialState.sorting) != null ? _table$initialState$s : []);
    };
    table.getPreSortedRowModel = () => table.getGroupedRowModel();
    table.getSortedRowModel = () => {
      if (!table._getSortedRowModel && table.options.getSortedRowModel) {
        table._getSortedRowModel = table.options.getSortedRowModel(table);
      }
      if (table.options.manualSorting || !table._getSortedRowModel) {
        return table.getPreSortedRowModel();
      }
      return table._getSortedRowModel();
    };
  }
};
const builtInFeatures = [
  Headers,
  ColumnVisibility,
  ColumnOrdering,
  ColumnPinning,
  ColumnFaceting,
  ColumnFiltering,
  GlobalFaceting,
  //depends on ColumnFaceting
  GlobalFiltering,
  //depends on ColumnFiltering
  RowSorting,
  ColumnGrouping,
  //depends on RowSorting
  RowExpanding,
  RowPagination,
  RowPinning,
  RowSelection,
  ColumnSizing
];
function createTable(options) {
  var _options$_features, _options$initialState;
  const _features = [...builtInFeatures, ...(_options$_features = options._features) != null ? _options$_features : []];
  let table = {
    _features
  };
  const defaultOptions = table._features.reduce((obj, feature) => {
    return Object.assign(obj, feature.getDefaultOptions == null ? void 0 : feature.getDefaultOptions(table));
  }, {});
  const mergeOptions = (options2) => {
    if (table.options.mergeOptions) {
      return table.options.mergeOptions(defaultOptions, options2);
    }
    return {
      ...defaultOptions,
      ...options2
    };
  };
  const coreInitialState = {};
  let initialState = {
    ...coreInitialState,
    ...(_options$initialState = options.initialState) != null ? _options$initialState : {}
  };
  table._features.forEach((feature) => {
    var _feature$getInitialSt;
    initialState = (_feature$getInitialSt = feature.getInitialState == null ? void 0 : feature.getInitialState(initialState)) != null ? _feature$getInitialSt : initialState;
  });
  const queued = [];
  let queuedTimeout = false;
  const coreInstance = {
    _features,
    options: {
      ...defaultOptions,
      ...options
    },
    initialState,
    _queue: (cb) => {
      queued.push(cb);
      if (!queuedTimeout) {
        queuedTimeout = true;
        Promise.resolve().then(() => {
          while (queued.length) {
            queued.shift()();
          }
          queuedTimeout = false;
        }).catch((error) => setTimeout(() => {
          throw error;
        }));
      }
    },
    reset: () => {
      table.setState(table.initialState);
    },
    setOptions: (updater) => {
      const newOptions = functionalUpdate(updater, table.options);
      table.options = mergeOptions(newOptions);
    },
    getState: () => {
      return table.options.state;
    },
    setState: (updater) => {
      table.options.onStateChange == null || table.options.onStateChange(updater);
    },
    _getRowId: (row, index, parent) => {
      var _table$options$getRow;
      return (_table$options$getRow = table.options.getRowId == null ? void 0 : table.options.getRowId(row, index, parent)) != null ? _table$options$getRow : `${parent ? [parent.id, index].join(".") : index}`;
    },
    getCoreRowModel: () => {
      if (!table._getCoreRowModel) {
        table._getCoreRowModel = table.options.getCoreRowModel(table);
      }
      return table._getCoreRowModel();
    },
    // The final calls start at the bottom of the model,
    // expanded rows, which then work their way up
    getRowModel: () => {
      return table.getPaginationRowModel();
    },
    //in next version, we should just pass in the row model as the optional 2nd arg
    getRow: (id, searchAll) => {
      let row = (searchAll ? table.getPrePaginationRowModel() : table.getRowModel()).rowsById[id];
      if (!row) {
        row = table.getCoreRowModel().rowsById[id];
        if (!row) {
          throw new Error();
        }
      }
      return row;
    },
    _getDefaultColumnDef: memo$1(() => [table.options.defaultColumn], (defaultColumn) => {
      var _defaultColumn;
      defaultColumn = (_defaultColumn = defaultColumn) != null ? _defaultColumn : {};
      return {
        header: (props) => {
          const resolvedColumnDef = props.header.column.columnDef;
          if (resolvedColumnDef.accessorKey) {
            return resolvedColumnDef.accessorKey;
          }
          if (resolvedColumnDef.accessorFn) {
            return resolvedColumnDef.id;
          }
          return null;
        },
        // footer: props => props.header.column.id,
        cell: (props) => {
          var _props$renderValue$to, _props$renderValue;
          return (_props$renderValue$to = (_props$renderValue = props.renderValue()) == null || _props$renderValue.toString == null ? void 0 : _props$renderValue.toString()) != null ? _props$renderValue$to : null;
        },
        ...table._features.reduce((obj, feature) => {
          return Object.assign(obj, feature.getDefaultColumnDef == null ? void 0 : feature.getDefaultColumnDef());
        }, {}),
        ...defaultColumn
      };
    }, getMemoOptions(options, "debugColumns")),
    _getColumnDefs: () => table.options.columns,
    getAllColumns: memo$1(() => [table._getColumnDefs()], (columnDefs) => {
      const recurseColumns = function(columnDefs2, parent, depth) {
        if (depth === void 0) {
          depth = 0;
        }
        return columnDefs2.map((columnDef) => {
          const column = createColumn(table, columnDef, depth, parent);
          const groupingColumnDef = columnDef;
          column.columns = groupingColumnDef.columns ? recurseColumns(groupingColumnDef.columns, column, depth + 1) : [];
          return column;
        });
      };
      return recurseColumns(columnDefs);
    }, getMemoOptions(options, "debugColumns")),
    getAllFlatColumns: memo$1(() => [table.getAllColumns()], (allColumns) => {
      return allColumns.flatMap((column) => {
        return column.getFlatColumns();
      });
    }, getMemoOptions(options, "debugColumns")),
    _getAllFlatColumnsById: memo$1(() => [table.getAllFlatColumns()], (flatColumns) => {
      return flatColumns.reduce((acc, column) => {
        acc[column.id] = column;
        return acc;
      }, {});
    }, getMemoOptions(options, "debugColumns")),
    getAllLeafColumns: memo$1(() => [table.getAllColumns(), table._getOrderColumnsFn()], (allColumns, orderColumns2) => {
      let leafColumns = allColumns.flatMap((column) => column.getLeafColumns());
      return orderColumns2(leafColumns);
    }, getMemoOptions(options, "debugColumns")),
    getColumn: (columnId) => {
      const column = table._getAllFlatColumnsById()[columnId];
      return column;
    }
  };
  Object.assign(table, coreInstance);
  for (let index = 0; index < table._features.length; index++) {
    const feature = table._features[index];
    feature == null || feature.createTable == null || feature.createTable(table);
  }
  return table;
}
function getCoreRowModel() {
  return (table) => memo$1(() => [table.options.data], (data) => {
    const rowModel = {
      rows: [],
      flatRows: [],
      rowsById: {}
    };
    const accessRows = function(originalRows, depth, parentRow) {
      if (depth === void 0) {
        depth = 0;
      }
      const rows = [];
      for (let i = 0; i < originalRows.length; i++) {
        const row = createRow(table, table._getRowId(originalRows[i], i, parentRow), originalRows[i], i, depth, void 0, parentRow == null ? void 0 : parentRow.id);
        rowModel.flatRows.push(row);
        rowModel.rowsById[row.id] = row;
        rows.push(row);
        if (table.options.getSubRows) {
          var _row$originalSubRows;
          row.originalSubRows = table.options.getSubRows(originalRows[i], i);
          if ((_row$originalSubRows = row.originalSubRows) != null && _row$originalSubRows.length) {
            row.subRows = accessRows(row.originalSubRows, depth + 1, row);
          }
        }
      }
      return rows;
    };
    rowModel.rows = accessRows(data);
    return rowModel;
  }, getMemoOptions(table.options, "debugTable", "getRowModel", () => table._autoResetPageIndex()));
}
function filterRows(rows, filterRowImpl, table) {
  if (table.options.filterFromLeafRows) {
    return filterRowModelFromLeafs(rows, filterRowImpl, table);
  }
  return filterRowModelFromRoot(rows, filterRowImpl, table);
}
function filterRowModelFromLeafs(rowsToFilter, filterRow, table) {
  var _table$options$maxLea;
  const newFilteredFlatRows = [];
  const newFilteredRowsById = {};
  const maxDepth = (_table$options$maxLea = table.options.maxLeafRowFilterDepth) != null ? _table$options$maxLea : 100;
  const recurseFilterRows = function(rowsToFilter2, depth) {
    if (depth === void 0) {
      depth = 0;
    }
    const rows = [];
    for (let i = 0; i < rowsToFilter2.length; i++) {
      var _row$subRows;
      let row = rowsToFilter2[i];
      const newRow = createRow(table, row.id, row.original, row.index, row.depth, void 0, row.parentId);
      newRow.columnFilters = row.columnFilters;
      if ((_row$subRows = row.subRows) != null && _row$subRows.length && depth < maxDepth) {
        newRow.subRows = recurseFilterRows(row.subRows, depth + 1);
        row = newRow;
        if (filterRow(row) && !newRow.subRows.length) {
          rows.push(row);
          newFilteredRowsById[row.id] = row;
          newFilteredFlatRows.push(row);
          continue;
        }
        if (filterRow(row) || newRow.subRows.length) {
          rows.push(row);
          newFilteredRowsById[row.id] = row;
          newFilteredFlatRows.push(row);
          continue;
        }
      } else {
        row = newRow;
        if (filterRow(row)) {
          rows.push(row);
          newFilteredRowsById[row.id] = row;
          newFilteredFlatRows.push(row);
        }
      }
    }
    return rows;
  };
  return {
    rows: recurseFilterRows(rowsToFilter),
    flatRows: newFilteredFlatRows,
    rowsById: newFilteredRowsById
  };
}
function filterRowModelFromRoot(rowsToFilter, filterRow, table) {
  var _table$options$maxLea2;
  const newFilteredFlatRows = [];
  const newFilteredRowsById = {};
  const maxDepth = (_table$options$maxLea2 = table.options.maxLeafRowFilterDepth) != null ? _table$options$maxLea2 : 100;
  const recurseFilterRows = function(rowsToFilter2, depth) {
    if (depth === void 0) {
      depth = 0;
    }
    const rows = [];
    for (let i = 0; i < rowsToFilter2.length; i++) {
      let row = rowsToFilter2[i];
      const pass = filterRow(row);
      if (pass) {
        var _row$subRows2;
        if ((_row$subRows2 = row.subRows) != null && _row$subRows2.length && depth < maxDepth) {
          const newRow = createRow(table, row.id, row.original, row.index, row.depth, void 0, row.parentId);
          newRow.subRows = recurseFilterRows(row.subRows, depth + 1);
          row = newRow;
        }
        rows.push(row);
        newFilteredFlatRows.push(row);
        newFilteredRowsById[row.id] = row;
      }
    }
    return rows;
  };
  return {
    rows: recurseFilterRows(rowsToFilter),
    flatRows: newFilteredFlatRows,
    rowsById: newFilteredRowsById
  };
}
function getFilteredRowModel() {
  return (table) => memo$1(() => [table.getPreFilteredRowModel(), table.getState().columnFilters, table.getState().globalFilter], (rowModel, columnFilters, globalFilter) => {
    if (!rowModel.rows.length || !(columnFilters != null && columnFilters.length) && !globalFilter) {
      for (let i = 0; i < rowModel.flatRows.length; i++) {
        rowModel.flatRows[i].columnFilters = {};
        rowModel.flatRows[i].columnFiltersMeta = {};
      }
      return rowModel;
    }
    const resolvedColumnFilters = [];
    const resolvedGlobalFilters = [];
    (columnFilters != null ? columnFilters : []).forEach((d) => {
      var _filterFn$resolveFilt;
      const column = table.getColumn(d.id);
      if (!column) {
        return;
      }
      const filterFn = column.getFilterFn();
      if (!filterFn) {
        return;
      }
      resolvedColumnFilters.push({
        id: d.id,
        filterFn,
        resolvedValue: (_filterFn$resolveFilt = filterFn.resolveFilterValue == null ? void 0 : filterFn.resolveFilterValue(d.value)) != null ? _filterFn$resolveFilt : d.value
      });
    });
    const filterableIds = (columnFilters != null ? columnFilters : []).map((d) => d.id);
    const globalFilterFn = table.getGlobalFilterFn();
    const globallyFilterableColumns = table.getAllLeafColumns().filter((column) => column.getCanGlobalFilter());
    if (globalFilter && globalFilterFn && globallyFilterableColumns.length) {
      filterableIds.push("__global__");
      globallyFilterableColumns.forEach((column) => {
        var _globalFilterFn$resol;
        resolvedGlobalFilters.push({
          id: column.id,
          filterFn: globalFilterFn,
          resolvedValue: (_globalFilterFn$resol = globalFilterFn.resolveFilterValue == null ? void 0 : globalFilterFn.resolveFilterValue(globalFilter)) != null ? _globalFilterFn$resol : globalFilter
        });
      });
    }
    let currentColumnFilter;
    let currentGlobalFilter;
    for (let j = 0; j < rowModel.flatRows.length; j++) {
      const row = rowModel.flatRows[j];
      row.columnFilters = {};
      if (resolvedColumnFilters.length) {
        for (let i = 0; i < resolvedColumnFilters.length; i++) {
          currentColumnFilter = resolvedColumnFilters[i];
          const id = currentColumnFilter.id;
          row.columnFilters[id] = currentColumnFilter.filterFn(row, id, currentColumnFilter.resolvedValue, (filterMeta) => {
            row.columnFiltersMeta[id] = filterMeta;
          });
        }
      }
      if (resolvedGlobalFilters.length) {
        for (let i = 0; i < resolvedGlobalFilters.length; i++) {
          currentGlobalFilter = resolvedGlobalFilters[i];
          const id = currentGlobalFilter.id;
          if (currentGlobalFilter.filterFn(row, id, currentGlobalFilter.resolvedValue, (filterMeta) => {
            row.columnFiltersMeta[id] = filterMeta;
          })) {
            row.columnFilters.__global__ = true;
            break;
          }
        }
        if (row.columnFilters.__global__ !== true) {
          row.columnFilters.__global__ = false;
        }
      }
    }
    const filterRowsImpl = (row) => {
      for (let i = 0; i < filterableIds.length; i++) {
        if (row.columnFilters[filterableIds[i]] === false) {
          return false;
        }
      }
      return true;
    };
    return filterRows(rowModel.rows, filterRowsImpl, table);
  }, getMemoOptions(table.options, "debugTable", "getFilteredRowModel", () => table._autoResetPageIndex()));
}
function getSortedRowModel() {
  return (table) => memo$1(() => [table.getState().sorting, table.getPreSortedRowModel()], (sorting, rowModel) => {
    if (!rowModel.rows.length || !(sorting != null && sorting.length)) {
      return rowModel;
    }
    const sortingState = table.getState().sorting;
    const sortedFlatRows = [];
    const availableSorting = sortingState.filter((sort) => {
      var _table$getColumn;
      return (_table$getColumn = table.getColumn(sort.id)) == null ? void 0 : _table$getColumn.getCanSort();
    });
    const columnInfoById = {};
    availableSorting.forEach((sortEntry) => {
      const column = table.getColumn(sortEntry.id);
      if (!column) return;
      columnInfoById[sortEntry.id] = {
        sortUndefined: column.columnDef.sortUndefined,
        invertSorting: column.columnDef.invertSorting,
        sortingFn: column.getSortingFn()
      };
    });
    const sortData = (rows) => {
      const sortedData = rows.map((row) => ({
        ...row
      }));
      sortedData.sort((rowA, rowB) => {
        for (let i = 0; i < availableSorting.length; i += 1) {
          var _sortEntry$desc;
          const sortEntry = availableSorting[i];
          const columnInfo = columnInfoById[sortEntry.id];
          const sortUndefined = columnInfo.sortUndefined;
          const isDesc = (_sortEntry$desc = sortEntry == null ? void 0 : sortEntry.desc) != null ? _sortEntry$desc : false;
          let sortInt = 0;
          if (sortUndefined) {
            const aValue = rowA.getValue(sortEntry.id);
            const bValue = rowB.getValue(sortEntry.id);
            const aUndefined = aValue === void 0;
            const bUndefined = bValue === void 0;
            if (aUndefined || bUndefined) {
              if (sortUndefined === "first") return aUndefined ? -1 : 1;
              if (sortUndefined === "last") return aUndefined ? 1 : -1;
              sortInt = aUndefined && bUndefined ? 0 : aUndefined ? sortUndefined : -sortUndefined;
            }
          }
          if (sortInt === 0) {
            sortInt = columnInfo.sortingFn(rowA, rowB, sortEntry.id);
          }
          if (sortInt !== 0) {
            if (isDesc) {
              sortInt *= -1;
            }
            if (columnInfo.invertSorting) {
              sortInt *= -1;
            }
            return sortInt;
          }
        }
        return rowA.index - rowB.index;
      });
      sortedData.forEach((row) => {
        var _row$subRows;
        sortedFlatRows.push(row);
        if ((_row$subRows = row.subRows) != null && _row$subRows.length) {
          row.subRows = sortData(row.subRows);
        }
      });
      return sortedData;
    };
    return {
      rows: sortData(rowModel.rows),
      flatRows: sortedFlatRows,
      rowsById: rowModel.rowsById
    };
  }, getMemoOptions(table.options, "debugTable", "getSortedRowModel", () => table._autoResetPageIndex()));
}

/* table.svelte.ts generated by Svelte v5.48.0 */

function mergeObjects(...sources) {
	const target = {};

	for (let i = 0; i < sources.length; i++) {
		let source = sources[i];

		if (typeof source === "function") source = source();

		if (source) {
			const descriptors = Object.getOwnPropertyDescriptors(source);

			for (const key in descriptors) {
				if (key in target) continue;

				Object.defineProperty(target, key, {
					enumerable: true,
					get() {
						for (let j = sources.length - 1; j >= 0; j--) {
							let s = sources[j];

							if (typeof s === "function") s = s();

							const v = (s || {})[key];

							if (v !== void 0) return v;
						}
					}
				});
			}
		}
	}

	return target;
}

function createSvelteTable(options) {
	const resolvedOptions = mergeObjects(
		{
			state: {},
			onStateChange() {},
			renderFallbackValue: null,
			mergeOptions: (defaultOptions, opts) => {
				return mergeObjects(defaultOptions, opts);
			}
		},
		options
	);

	const table = createTable(resolvedOptions);
	let state$1 = state(proxy(table.initialState));
	let version = state(0);

	function updateOptions() {
		table.setOptions((prev) => {
			return mergeObjects(prev, options, {
				state: mergeObjects(get(state$1), options.state || {}),
				onStateChange: (updater) => {
					if (updater instanceof Function) set(state$1, updater(get(state$1)), true); else set(state$1, mergeObjects(get(state$1), updater), true);

					set(version, get(version) + 1);
					options.onStateChange?.(updater);
				}
			});
		});
	}

	updateOptions();

	user_pre_effect(() => {
		updateOptions();
	});

	return {
		getRowModel: () => {
			void get(version);

			return table.getRowModel();
		},

		getHeaderGroups: () => {
			void get(version);

			return table.getHeaderGroups();
		},

		getColumn: (id) => {
			void get(version);

			return table.getColumn(id);
		}
	};
}

function memo(getDeps, fn, opts) {
  let deps = opts.initialDeps ?? [];
  let result;
  let isInitial = true;
  function memoizedFunction() {
    var _a, _b, _c;
    let depTime;
    if (opts.key && ((_a = opts.debug) == null ? void 0 : _a.call(opts))) depTime = Date.now();
    const newDeps = getDeps();
    const depsChanged = newDeps.length !== deps.length || newDeps.some((dep, index) => deps[index] !== dep);
    if (!depsChanged) {
      return result;
    }
    deps = newDeps;
    let resultTime;
    if (opts.key && ((_b = opts.debug) == null ? void 0 : _b.call(opts))) resultTime = Date.now();
    result = fn(...newDeps);
    if (opts.key && ((_c = opts.debug) == null ? void 0 : _c.call(opts))) {
      const depEndTime = Math.round((Date.now() - depTime) * 100) / 100;
      const resultEndTime = Math.round((Date.now() - resultTime) * 100) / 100;
      const resultFpsPercentage = resultEndTime / 16;
      const pad = (str, num) => {
        str = String(str);
        while (str.length < num) {
          str = " " + str;
        }
        return str;
      };
      console.info(
        `%c⏱ ${pad(resultEndTime, 5)} /${pad(depEndTime, 5)} ms`,
        `
            font-size: .6rem;
            font-weight: bold;
            color: hsl(${Math.max(
          0,
          Math.min(120 - 120 * resultFpsPercentage, 120)
        )}deg 100% 31%);`,
        opts == null ? void 0 : opts.key
      );
    }
    if ((opts == null ? void 0 : opts.onChange) && !(isInitial && opts.skipInitialOnChange)) {
      opts.onChange(result);
    }
    isInitial = false;
    return result;
  }
  memoizedFunction.updateDeps = (newDeps) => {
    deps = newDeps;
  };
  return memoizedFunction;
}
function notUndefined(value, msg) {
  if (value === void 0) {
    throw new Error(`Unexpected undefined${""}`);
  } else {
    return value;
  }
}
const approxEqual = (a, b) => Math.abs(a - b) < 1.01;
const debounce = (targetWindow, fn, ms) => {
  let timeoutId;
  return function(...args) {
    targetWindow.clearTimeout(timeoutId);
    timeoutId = targetWindow.setTimeout(() => fn.apply(this, args), ms);
  };
};

const getRect = (element) => {
  const { offsetWidth, offsetHeight } = element;
  return { width: offsetWidth, height: offsetHeight };
};
const defaultKeyExtractor = (index) => index;
const defaultRangeExtractor = (range) => {
  const start = Math.max(range.startIndex - range.overscan, 0);
  const end = Math.min(range.endIndex + range.overscan, range.count - 1);
  const arr = [];
  for (let i = start; i <= end; i++) {
    arr.push(i);
  }
  return arr;
};
const observeElementRect = (instance, cb) => {
  const element = instance.scrollElement;
  if (!element) {
    return;
  }
  const targetWindow = instance.targetWindow;
  if (!targetWindow) {
    return;
  }
  const handler = (rect) => {
    const { width, height } = rect;
    cb({ width: Math.round(width), height: Math.round(height) });
  };
  handler(getRect(element));
  if (!targetWindow.ResizeObserver) {
    return () => {
    };
  }
  const observer = new targetWindow.ResizeObserver((entries) => {
    const run = () => {
      const entry = entries[0];
      if (entry == null ? void 0 : entry.borderBoxSize) {
        const box = entry.borderBoxSize[0];
        if (box) {
          handler({ width: box.inlineSize, height: box.blockSize });
          return;
        }
      }
      handler(getRect(element));
    };
    instance.options.useAnimationFrameWithResizeObserver ? requestAnimationFrame(run) : run();
  });
  observer.observe(element, { box: "border-box" });
  return () => {
    observer.unobserve(element);
  };
};
const addEventListenerOptions = {
  passive: true
};
const supportsScrollend = typeof window == "undefined" ? true : "onscrollend" in window;
const observeElementOffset = (instance, cb) => {
  const element = instance.scrollElement;
  if (!element) {
    return;
  }
  const targetWindow = instance.targetWindow;
  if (!targetWindow) {
    return;
  }
  let offset = 0;
  const fallback = instance.options.useScrollendEvent && supportsScrollend ? () => void 0 : debounce(
    targetWindow,
    () => {
      cb(offset, false);
    },
    instance.options.isScrollingResetDelay
  );
  const createHandler = (isScrolling) => () => {
    const { horizontal, isRtl } = instance.options;
    offset = horizontal ? element["scrollLeft"] * (isRtl && -1 || 1) : element["scrollTop"];
    fallback();
    cb(offset, isScrolling);
  };
  const handler = createHandler(true);
  const endHandler = createHandler(false);
  element.addEventListener("scroll", handler, addEventListenerOptions);
  const registerScrollendEvent = instance.options.useScrollendEvent && supportsScrollend;
  if (registerScrollendEvent) {
    element.addEventListener("scrollend", endHandler, addEventListenerOptions);
  }
  return () => {
    element.removeEventListener("scroll", handler);
    if (registerScrollendEvent) {
      element.removeEventListener("scrollend", endHandler);
    }
  };
};
const measureElement = (element, entry, instance) => {
  if (entry == null ? void 0 : entry.borderBoxSize) {
    const box = entry.borderBoxSize[0];
    if (box) {
      const size = Math.round(
        box[instance.options.horizontal ? "inlineSize" : "blockSize"]
      );
      return size;
    }
  }
  return element[instance.options.horizontal ? "offsetWidth" : "offsetHeight"];
};
const elementScroll = (offset, {
  adjustments = 0,
  behavior
}, instance) => {
  var _a, _b;
  const toOffset = offset + adjustments;
  (_b = (_a = instance.scrollElement) == null ? void 0 : _a.scrollTo) == null ? void 0 : _b.call(_a, {
    [instance.options.horizontal ? "left" : "top"]: toOffset,
    behavior
  });
};
class Virtualizer {
  constructor(opts) {
    this.unsubs = [];
    this.scrollElement = null;
    this.targetWindow = null;
    this.isScrolling = false;
    this.scrollState = null;
    this.measurementsCache = [];
    this.itemSizeCache = /* @__PURE__ */ new Map();
    this.laneAssignments = /* @__PURE__ */ new Map();
    this.pendingMeasuredCacheIndexes = [];
    this.prevLanes = void 0;
    this.lanesChangedFlag = false;
    this.lanesSettling = false;
    this.scrollRect = null;
    this.scrollOffset = null;
    this.scrollDirection = null;
    this.scrollAdjustments = 0;
    this.elementsCache = /* @__PURE__ */ new Map();
    this.now = () => {
      var _a, _b, _c;
      return ((_c = (_b = (_a = this.targetWindow) == null ? void 0 : _a.performance) == null ? void 0 : _b.now) == null ? void 0 : _c.call(_b)) ?? Date.now();
    };
    this.observer = /* @__PURE__ */ (() => {
      let _ro = null;
      const get = () => {
        if (_ro) {
          return _ro;
        }
        if (!this.targetWindow || !this.targetWindow.ResizeObserver) {
          return null;
        }
        return _ro = new this.targetWindow.ResizeObserver((entries) => {
          entries.forEach((entry) => {
            const run = () => {
              const node = entry.target;
              const index = this.indexFromElement(node);
              if (!node.isConnected) {
                this.observer.unobserve(node);
                return;
              }
              if (this.shouldMeasureDuringScroll(index)) {
                this.resizeItem(
                  index,
                  this.options.measureElement(node, entry, this)
                );
              }
            };
            this.options.useAnimationFrameWithResizeObserver ? requestAnimationFrame(run) : run();
          });
        });
      };
      return {
        disconnect: () => {
          var _a;
          (_a = get()) == null ? void 0 : _a.disconnect();
          _ro = null;
        },
        observe: (target) => {
          var _a;
          return (_a = get()) == null ? void 0 : _a.observe(target, { box: "border-box" });
        },
        unobserve: (target) => {
          var _a;
          return (_a = get()) == null ? void 0 : _a.unobserve(target);
        }
      };
    })();
    this.range = null;
    this.setOptions = (opts2) => {
      Object.entries(opts2).forEach(([key, value]) => {
        if (typeof value === "undefined") delete opts2[key];
      });
      this.options = {
        debug: false,
        initialOffset: 0,
        overscan: 1,
        paddingStart: 0,
        paddingEnd: 0,
        scrollPaddingStart: 0,
        scrollPaddingEnd: 0,
        horizontal: false,
        getItemKey: defaultKeyExtractor,
        rangeExtractor: defaultRangeExtractor,
        onChange: () => {
        },
        measureElement,
        initialRect: { width: 0, height: 0 },
        scrollMargin: 0,
        gap: 0,
        indexAttribute: "data-index",
        initialMeasurementsCache: [],
        lanes: 1,
        isScrollingResetDelay: 150,
        enabled: true,
        isRtl: false,
        useScrollendEvent: false,
        useAnimationFrameWithResizeObserver: false,
        ...opts2
      };
    };
    this.notify = (sync) => {
      var _a, _b;
      (_b = (_a = this.options).onChange) == null ? void 0 : _b.call(_a, this, sync);
    };
    this.maybeNotify = memo(
      () => {
        this.calculateRange();
        return [
          this.isScrolling,
          this.range ? this.range.startIndex : null,
          this.range ? this.range.endIndex : null
        ];
      },
      (isScrolling) => {
        this.notify(isScrolling);
      },
      {
        key: false,
        debug: () => this.options.debug,
        initialDeps: [
          this.isScrolling,
          this.range ? this.range.startIndex : null,
          this.range ? this.range.endIndex : null
        ]
      }
    );
    this.cleanup = () => {
      this.unsubs.filter(Boolean).forEach((d) => d());
      this.unsubs = [];
      this.observer.disconnect();
      if (this.rafId != null && this.targetWindow) {
        this.targetWindow.cancelAnimationFrame(this.rafId);
        this.rafId = null;
      }
      this.scrollState = null;
      this.scrollElement = null;
      this.targetWindow = null;
    };
    this._didMount = () => {
      return () => {
        this.cleanup();
      };
    };
    this._willUpdate = () => {
      var _a;
      const scrollElement = this.options.enabled ? this.options.getScrollElement() : null;
      if (this.scrollElement !== scrollElement) {
        this.cleanup();
        if (!scrollElement) {
          this.maybeNotify();
          return;
        }
        this.scrollElement = scrollElement;
        if (this.scrollElement && "ownerDocument" in this.scrollElement) {
          this.targetWindow = this.scrollElement.ownerDocument.defaultView;
        } else {
          this.targetWindow = ((_a = this.scrollElement) == null ? void 0 : _a.window) ?? null;
        }
        this.elementsCache.forEach((cached) => {
          this.observer.observe(cached);
        });
        this.unsubs.push(
          this.options.observeElementRect(this, (rect) => {
            this.scrollRect = rect;
            this.maybeNotify();
          })
        );
        this.unsubs.push(
          this.options.observeElementOffset(this, (offset, isScrolling) => {
            this.scrollAdjustments = 0;
            this.scrollDirection = isScrolling ? this.getScrollOffset() < offset ? "forward" : "backward" : null;
            this.scrollOffset = offset;
            this.isScrolling = isScrolling;
            if (this.scrollState) {
              this.scheduleScrollReconcile();
            }
            this.maybeNotify();
          })
        );
        this._scrollToOffset(this.getScrollOffset(), {
          adjustments: void 0,
          behavior: void 0
        });
      }
    };
    this.rafId = null;
    this.getSize = () => {
      if (!this.options.enabled) {
        this.scrollRect = null;
        return 0;
      }
      this.scrollRect = this.scrollRect ?? this.options.initialRect;
      return this.scrollRect[this.options.horizontal ? "width" : "height"];
    };
    this.getScrollOffset = () => {
      if (!this.options.enabled) {
        this.scrollOffset = null;
        return 0;
      }
      this.scrollOffset = this.scrollOffset ?? (typeof this.options.initialOffset === "function" ? this.options.initialOffset() : this.options.initialOffset);
      return this.scrollOffset;
    };
    this.getFurthestMeasurement = (measurements, index) => {
      const furthestMeasurementsFound = /* @__PURE__ */ new Map();
      const furthestMeasurements = /* @__PURE__ */ new Map();
      for (let m = index - 1; m >= 0; m--) {
        const measurement = measurements[m];
        if (furthestMeasurementsFound.has(measurement.lane)) {
          continue;
        }
        const previousFurthestMeasurement = furthestMeasurements.get(
          measurement.lane
        );
        if (previousFurthestMeasurement == null || measurement.end > previousFurthestMeasurement.end) {
          furthestMeasurements.set(measurement.lane, measurement);
        } else if (measurement.end < previousFurthestMeasurement.end) {
          furthestMeasurementsFound.set(measurement.lane, true);
        }
        if (furthestMeasurementsFound.size === this.options.lanes) {
          break;
        }
      }
      return furthestMeasurements.size === this.options.lanes ? Array.from(furthestMeasurements.values()).sort((a, b) => {
        if (a.end === b.end) {
          return a.index - b.index;
        }
        return a.end - b.end;
      })[0] : void 0;
    };
    this.getMeasurementOptions = memo(
      () => [
        this.options.count,
        this.options.paddingStart,
        this.options.scrollMargin,
        this.options.getItemKey,
        this.options.enabled,
        this.options.lanes
      ],
      (count, paddingStart, scrollMargin, getItemKey, enabled, lanes) => {
        const lanesChanged = this.prevLanes !== void 0 && this.prevLanes !== lanes;
        if (lanesChanged) {
          this.lanesChangedFlag = true;
        }
        this.prevLanes = lanes;
        this.pendingMeasuredCacheIndexes = [];
        return {
          count,
          paddingStart,
          scrollMargin,
          getItemKey,
          enabled,
          lanes
        };
      },
      {
        key: false
      }
    );
    this.getMeasurements = memo(
      () => [this.getMeasurementOptions(), this.itemSizeCache],
      ({ count, paddingStart, scrollMargin, getItemKey, enabled, lanes }, itemSizeCache) => {
        if (!enabled) {
          this.measurementsCache = [];
          this.itemSizeCache.clear();
          this.laneAssignments.clear();
          return [];
        }
        if (this.laneAssignments.size > count) {
          for (const index of this.laneAssignments.keys()) {
            if (index >= count) {
              this.laneAssignments.delete(index);
            }
          }
        }
        if (this.lanesChangedFlag) {
          this.lanesChangedFlag = false;
          this.lanesSettling = true;
          this.measurementsCache = [];
          this.itemSizeCache.clear();
          this.laneAssignments.clear();
          this.pendingMeasuredCacheIndexes = [];
        }
        if (this.measurementsCache.length === 0 && !this.lanesSettling) {
          this.measurementsCache = this.options.initialMeasurementsCache;
          this.measurementsCache.forEach((item) => {
            this.itemSizeCache.set(item.key, item.size);
          });
        }
        const min = this.lanesSettling ? 0 : this.pendingMeasuredCacheIndexes.length > 0 ? Math.min(...this.pendingMeasuredCacheIndexes) : 0;
        this.pendingMeasuredCacheIndexes = [];
        if (this.lanesSettling && this.measurementsCache.length === count) {
          this.lanesSettling = false;
        }
        const measurements = this.measurementsCache.slice(0, min);
        const laneLastIndex = new Array(lanes).fill(
          void 0
        );
        for (let m = 0; m < min; m++) {
          const item = measurements[m];
          if (item) {
            laneLastIndex[item.lane] = m;
          }
        }
        for (let i = min; i < count; i++) {
          const key = getItemKey(i);
          const cachedLane = this.laneAssignments.get(i);
          let lane;
          let start;
          if (cachedLane !== void 0 && this.options.lanes > 1) {
            lane = cachedLane;
            const prevIndex = laneLastIndex[lane];
            const prevInLane = prevIndex !== void 0 ? measurements[prevIndex] : void 0;
            start = prevInLane ? prevInLane.end + this.options.gap : paddingStart + scrollMargin;
          } else {
            const furthestMeasurement = this.options.lanes === 1 ? measurements[i - 1] : this.getFurthestMeasurement(measurements, i);
            start = furthestMeasurement ? furthestMeasurement.end + this.options.gap : paddingStart + scrollMargin;
            lane = furthestMeasurement ? furthestMeasurement.lane : i % this.options.lanes;
            if (this.options.lanes > 1) {
              this.laneAssignments.set(i, lane);
            }
          }
          const measuredSize = itemSizeCache.get(key);
          const size = typeof measuredSize === "number" ? measuredSize : this.options.estimateSize(i);
          const end = start + size;
          measurements[i] = {
            index: i,
            start,
            size,
            end,
            key,
            lane
          };
          laneLastIndex[lane] = i;
        }
        this.measurementsCache = measurements;
        return measurements;
      },
      {
        key: false,
        debug: () => this.options.debug
      }
    );
    this.calculateRange = memo(
      () => [
        this.getMeasurements(),
        this.getSize(),
        this.getScrollOffset(),
        this.options.lanes
      ],
      (measurements, outerSize, scrollOffset, lanes) => {
        return this.range = measurements.length > 0 && outerSize > 0 ? calculateRange({
          measurements,
          outerSize,
          scrollOffset,
          lanes
        }) : null;
      },
      {
        key: false,
        debug: () => this.options.debug
      }
    );
    this.getVirtualIndexes = memo(
      () => {
        let startIndex = null;
        let endIndex = null;
        const range = this.calculateRange();
        if (range) {
          startIndex = range.startIndex;
          endIndex = range.endIndex;
        }
        this.maybeNotify.updateDeps([this.isScrolling, startIndex, endIndex]);
        return [
          this.options.rangeExtractor,
          this.options.overscan,
          this.options.count,
          startIndex,
          endIndex
        ];
      },
      (rangeExtractor, overscan, count, startIndex, endIndex) => {
        return startIndex === null || endIndex === null ? [] : rangeExtractor({
          startIndex,
          endIndex,
          overscan,
          count
        });
      },
      {
        key: false,
        debug: () => this.options.debug
      }
    );
    this.indexFromElement = (node) => {
      const attributeName = this.options.indexAttribute;
      const indexStr = node.getAttribute(attributeName);
      if (!indexStr) {
        console.warn(
          `Missing attribute name '${attributeName}={index}' on measured element.`
        );
        return -1;
      }
      return parseInt(indexStr, 10);
    };
    this.shouldMeasureDuringScroll = (index) => {
      var _a;
      if (!this.scrollState || this.scrollState.behavior !== "smooth") {
        return true;
      }
      const scrollIndex = this.scrollState.index ?? ((_a = this.getVirtualItemForOffset(this.scrollState.lastTargetOffset)) == null ? void 0 : _a.index);
      if (scrollIndex !== void 0 && this.range) {
        const bufferSize = Math.max(
          this.options.overscan,
          Math.ceil((this.range.endIndex - this.range.startIndex) / 2)
        );
        const minIndex = Math.max(0, scrollIndex - bufferSize);
        const maxIndex = Math.min(
          this.options.count - 1,
          scrollIndex + bufferSize
        );
        return index >= minIndex && index <= maxIndex;
      }
      return true;
    };
    this.measureElement = (node) => {
      if (!node) {
        this.elementsCache.forEach((cached, key2) => {
          if (!cached.isConnected) {
            this.observer.unobserve(cached);
            this.elementsCache.delete(key2);
          }
        });
        return;
      }
      const index = this.indexFromElement(node);
      const key = this.options.getItemKey(index);
      const prevNode = this.elementsCache.get(key);
      if (prevNode !== node) {
        if (prevNode) {
          this.observer.unobserve(prevNode);
        }
        this.observer.observe(node);
        this.elementsCache.set(key, node);
      }
      if ((!this.isScrolling || this.scrollState) && this.shouldMeasureDuringScroll(index)) {
        this.resizeItem(index, this.options.measureElement(node, void 0, this));
      }
    };
    this.resizeItem = (index, size) => {
      var _a;
      const item = this.measurementsCache[index];
      if (!item) return;
      const itemSize = this.itemSizeCache.get(item.key) ?? item.size;
      const delta = size - itemSize;
      if (delta !== 0) {
        if (((_a = this.scrollState) == null ? void 0 : _a.behavior) !== "smooth" && (this.shouldAdjustScrollPositionOnItemSizeChange !== void 0 ? this.shouldAdjustScrollPositionOnItemSizeChange(item, delta, this) : item.start < this.getScrollOffset() + this.scrollAdjustments)) {
          this._scrollToOffset(this.getScrollOffset(), {
            adjustments: this.scrollAdjustments += delta,
            behavior: void 0
          });
        }
        this.pendingMeasuredCacheIndexes.push(item.index);
        this.itemSizeCache = new Map(this.itemSizeCache.set(item.key, size));
        this.notify(false);
      }
    };
    this.getVirtualItems = memo(
      () => [this.getVirtualIndexes(), this.getMeasurements()],
      (indexes, measurements) => {
        const virtualItems = [];
        for (let k = 0, len = indexes.length; k < len; k++) {
          const i = indexes[k];
          const measurement = measurements[i];
          virtualItems.push(measurement);
        }
        return virtualItems;
      },
      {
        key: false,
        debug: () => this.options.debug
      }
    );
    this.getVirtualItemForOffset = (offset) => {
      const measurements = this.getMeasurements();
      if (measurements.length === 0) {
        return void 0;
      }
      return notUndefined(
        measurements[findNearestBinarySearch(
          0,
          measurements.length - 1,
          (index) => notUndefined(measurements[index]).start,
          offset
        )]
      );
    };
    this.getMaxScrollOffset = () => {
      if (!this.scrollElement) return 0;
      if ("scrollHeight" in this.scrollElement) {
        return this.options.horizontal ? this.scrollElement.scrollWidth - this.scrollElement.clientWidth : this.scrollElement.scrollHeight - this.scrollElement.clientHeight;
      } else {
        const doc = this.scrollElement.document.documentElement;
        return this.options.horizontal ? doc.scrollWidth - this.scrollElement.innerWidth : doc.scrollHeight - this.scrollElement.innerHeight;
      }
    };
    this.getOffsetForAlignment = (toOffset, align, itemSize = 0) => {
      if (!this.scrollElement) return 0;
      const size = this.getSize();
      const scrollOffset = this.getScrollOffset();
      if (align === "auto") {
        align = toOffset >= scrollOffset + size ? "end" : "start";
      }
      if (align === "center") {
        toOffset += (itemSize - size) / 2;
      } else if (align === "end") {
        toOffset -= size;
      }
      const maxOffset = this.getMaxScrollOffset();
      return Math.max(Math.min(maxOffset, toOffset), 0);
    };
    this.getOffsetForIndex = (index, align = "auto") => {
      index = Math.max(0, Math.min(index, this.options.count - 1));
      const size = this.getSize();
      const scrollOffset = this.getScrollOffset();
      const item = this.measurementsCache[index];
      if (!item) return;
      if (align === "auto") {
        if (item.end >= scrollOffset + size - this.options.scrollPaddingEnd) {
          align = "end";
        } else if (item.start <= scrollOffset + this.options.scrollPaddingStart) {
          align = "start";
        } else {
          return [scrollOffset, align];
        }
      }
      if (align === "end" && index === this.options.count - 1) {
        return [this.getMaxScrollOffset(), align];
      }
      const toOffset = align === "end" ? item.end + this.options.scrollPaddingEnd : item.start - this.options.scrollPaddingStart;
      return [
        this.getOffsetForAlignment(toOffset, align, item.size),
        align
      ];
    };
    this.scrollToOffset = (toOffset, { align = "start", behavior = "auto" } = {}) => {
      const offset = this.getOffsetForAlignment(toOffset, align);
      const now = this.now();
      this.scrollState = {
        index: null,
        align,
        behavior,
        startedAt: now,
        lastTargetOffset: offset,
        stableFrames: 0
      };
      this._scrollToOffset(offset, { adjustments: void 0, behavior });
      this.scheduleScrollReconcile();
    };
    this.scrollToIndex = (index, {
      align: initialAlign = "auto",
      behavior = "auto"
    } = {}) => {
      index = Math.max(0, Math.min(index, this.options.count - 1));
      const offsetInfo = this.getOffsetForIndex(index, initialAlign);
      if (!offsetInfo) {
        return;
      }
      const [offset, align] = offsetInfo;
      const now = this.now();
      this.scrollState = {
        index,
        align,
        behavior,
        startedAt: now,
        lastTargetOffset: offset,
        stableFrames: 0
      };
      this._scrollToOffset(offset, { adjustments: void 0, behavior });
      this.scheduleScrollReconcile();
    };
    this.scrollBy = (delta, { behavior = "auto" } = {}) => {
      const offset = this.getScrollOffset() + delta;
      const now = this.now();
      this.scrollState = {
        index: null,
        align: "start",
        behavior,
        startedAt: now,
        lastTargetOffset: offset,
        stableFrames: 0
      };
      this._scrollToOffset(offset, { adjustments: void 0, behavior });
      this.scheduleScrollReconcile();
    };
    this.getTotalSize = () => {
      var _a;
      const measurements = this.getMeasurements();
      let end;
      if (measurements.length === 0) {
        end = this.options.paddingStart;
      } else if (this.options.lanes === 1) {
        end = ((_a = measurements[measurements.length - 1]) == null ? void 0 : _a.end) ?? 0;
      } else {
        const endByLane = Array(this.options.lanes).fill(null);
        let endIndex = measurements.length - 1;
        while (endIndex >= 0 && endByLane.some((val) => val === null)) {
          const item = measurements[endIndex];
          if (endByLane[item.lane] === null) {
            endByLane[item.lane] = item.end;
          }
          endIndex--;
        }
        end = Math.max(...endByLane.filter((val) => val !== null));
      }
      return Math.max(
        end - this.options.scrollMargin + this.options.paddingEnd,
        0
      );
    };
    this._scrollToOffset = (offset, {
      adjustments,
      behavior
    }) => {
      this.options.scrollToFn(offset, { behavior, adjustments }, this);
    };
    this.measure = () => {
      this.itemSizeCache = /* @__PURE__ */ new Map();
      this.laneAssignments = /* @__PURE__ */ new Map();
      this.notify(false);
    };
    this.setOptions(opts);
  }
  scheduleScrollReconcile() {
    if (!this.targetWindow) {
      this.scrollState = null;
      return;
    }
    if (this.rafId != null) return;
    this.rafId = this.targetWindow.requestAnimationFrame(() => {
      this.rafId = null;
      this.reconcileScroll();
    });
  }
  reconcileScroll() {
    if (!this.scrollState) return;
    const el = this.scrollElement;
    if (!el) return;
    const MAX_RECONCILE_MS = 5e3;
    if (this.now() - this.scrollState.startedAt > MAX_RECONCILE_MS) {
      this.scrollState = null;
      return;
    }
    const offsetInfo = this.scrollState.index != null ? this.getOffsetForIndex(this.scrollState.index, this.scrollState.align) : void 0;
    const targetOffset = offsetInfo ? offsetInfo[0] : this.scrollState.lastTargetOffset;
    const STABLE_FRAMES = 1;
    const targetChanged = targetOffset !== this.scrollState.lastTargetOffset;
    if (!targetChanged && approxEqual(targetOffset, this.getScrollOffset())) {
      this.scrollState.stableFrames++;
      if (this.scrollState.stableFrames >= STABLE_FRAMES) {
        this.scrollState = null;
        return;
      }
    } else {
      this.scrollState.stableFrames = 0;
      if (targetChanged) {
        this.scrollState.lastTargetOffset = targetOffset;
        this.scrollState.behavior = "auto";
        this._scrollToOffset(targetOffset, {
          adjustments: void 0,
          behavior: "auto"
        });
      }
    }
    this.scheduleScrollReconcile();
  }
}
const findNearestBinarySearch = (low, high, getCurrentValue, value) => {
  while (low <= high) {
    const middle = (low + high) / 2 | 0;
    const currentValue = getCurrentValue(middle);
    if (currentValue < value) {
      low = middle + 1;
    } else if (currentValue > value) {
      high = middle - 1;
    } else {
      return middle;
    }
  }
  if (low > 0) {
    return low - 1;
  } else {
    return 0;
  }
};
function calculateRange({
  measurements,
  outerSize,
  scrollOffset,
  lanes
}) {
  const lastIndex = measurements.length - 1;
  const getOffset = (index) => measurements[index].start;
  if (measurements.length <= lanes) {
    return {
      startIndex: 0,
      endIndex: lastIndex
    };
  }
  let startIndex = findNearestBinarySearch(
    0,
    lastIndex,
    getOffset,
    scrollOffset
  );
  let endIndex = startIndex;
  if (lanes === 1) {
    while (endIndex < lastIndex && measurements[endIndex].end < scrollOffset + outerSize) {
      endIndex++;
    }
  } else if (lanes > 1) {
    const endPerLane = Array(lanes).fill(0);
    while (endIndex < lastIndex && endPerLane.some((pos) => pos < scrollOffset + outerSize)) {
      const item = measurements[endIndex];
      endPerLane[item.lane] = item.end;
      endIndex++;
    }
    const startPerLane = Array(lanes).fill(scrollOffset + outerSize);
    while (startIndex >= 0 && startPerLane.some((pos) => pos >= scrollOffset)) {
      const item = measurements[startIndex];
      startPerLane[item.lane] = item.start;
      startIndex--;
    }
    startIndex = Math.max(0, startIndex - startIndex % lanes);
    endIndex = Math.min(lastIndex, endIndex + (lanes - 1 - endIndex % lanes));
  }
  return { startIndex, endIndex };
}

/* virtual.svelte.ts generated by Svelte v5.48.0 */

function createSvelteVirtualizer(options) {
	let version = state(0);

	const virtualizer = new Virtualizer({
		observeElementRect,
		observeElementOffset,
		scrollToFn: elementScroll,
		...options,
		onChange: (instance, sync) => {
			if (sync) {
				set(version, get(version) + 1);
			} else {
				queueMicrotask(() => {
					set(version, get(version) + 1);
				});
			}

			options.onChange?.(instance, sync);
		}
	});

	user_effect(() => {
		const cleanup = virtualizer._didMount();

		untrack(() => {
			set(version, get(version) + 1);
		});

		return cleanup;
	});

	let prev_count = 0;

	user_effect(() => {
		const current_count = options.count;

		virtualizer.setOptions({
			observeElementRect,
			observeElementOffset,
			scrollToFn: elementScroll,
			...options,
			onChange: (instance, sync) => {
				if (sync) {
					set(version, get(version) + 1);
				} else {
					queueMicrotask(() => {
						set(version, get(version) + 1);
					});
				}

				options.onChange?.(instance, sync);
			}
		});

		if (prev_count === 0 && current_count > 0) {
			virtualizer.measure();
		}

		prev_count = current_count;
	});

	user_pre_effect(() => {
		void get(version);
		virtualizer._willUpdate();
	});

	return {
		instance: virtualizer,
		virtualItems: () => {
			void get(version);

			return virtualizer.getVirtualItems();
		},

		totalSize: () => {
			void get(version);

			return virtualizer.getTotalSize();
		}
	};
}

var root$d = from_html(`<button><span><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" class="svelte-fit6p6"><path d="m16.707 13.293-4-4a1 1 0 0 0-1.414 0l-4 4A1 1 0 0 0 8 15h8a1 1 0 0 0 .707-1.707z"></path></svg></span></button>`);

function SelectionButtons($$anchor, $$props) {
	push($$props, true);

	let on_click = prop($$props, 'on_click', 3, null);
	let is_first_position = user_derived(() => $$props.position === "column" ? $$props.coords[0] === 0 : $$props.coords[1] === 0);

	let direction = user_derived(() => $$props.position === "column"
		? get(is_first_position) ? "down" : "up"
		: get(is_first_position) ? "right" : "left");

	var button = root$d();

	button.__click = (e) => {
		e.stopPropagation();
		on_click() && on_click()();
	};

	var span = child(button);
	var svg = child(span);
	var path = child(svg);

	reset(svg);
	reset(span);
	reset(button);

	template_effect(() => {
		set_class(button, 1, `selection-button selection-button-${$$props.position ?? ''} ${get(is_first_position) ? `move-${get(direction)}` : ''}`, 'svelte-fit6p6');
		set_attribute(button, 'aria-label', `Select ${$$props.position}`);
		set_class(span, 1, clsx(get(direction)), 'svelte-fit6p6');
		set_attribute(path, 'data-name', get(direction));
	});

	append($$anchor, button);
	pop();
}

delegate(['click']);

var root$c = from_html(`<div class="bool-cell svelte-1d51ufy" role="button" tabindex="-1"><!></div>`);

function BooleanCell($$anchor, $$props) {
	push($$props, true);

	let value = prop($$props, 'value', 15, false),
		editable = prop($$props, 'editable', 3, true);

	function handle_change(val) {
		if (editable()) {
			$$props.on_change(val);
		}
	}

	var div = root$c();
	var node = child(div);

	Checkbox(node, {
		label: '',
		get interactive() {
			return editable();
		},
		on_change: handle_change,
		get value() {
			return value();
		},

		set value($$value) {
			value($$value);
		}
	});

	reset(div);
	append($$anchor, div);
	pop();
}

var root_1$7 = from_html(`<textarea tabindex="-1"></textarea>`);
var root_3$4 = from_html(`<span tabindex="0" role="button" placeholder=" "><!></span>`);
var root_10$1 = from_html(`<!> <!>`, 1);
var root$b = from_html(`<!> <!> <!>`, 1);

function EditableCell($$anchor, $$props) {
	push($$props, true);

	let value = prop($$props, 'value', 15, ""),
		display_value = prop($$props, 'display_value', 3, null),
		styling = prop($$props, 'styling', 3, ""),
		header = prop($$props, 'header', 3, false),
		datatype = prop($$props, 'datatype', 3, "str"),
		line_breaks = prop($$props, 'line_breaks', 3, true),
		editable = prop($$props, 'editable', 3, true),
		is_static = prop($$props, 'is_static', 3, false),
		max_chars = prop($$props, 'max_chars', 3, null),
		components = prop($$props, 'components', 19, () => ({})),
		is_dragging = prop($$props, 'is_dragging', 3, false),
		wrap_text = prop($$props, 'wrap_text', 3, false),
		show_selection_buttons = prop($$props, 'show_selection_buttons', 3, false),
		on_select_column = prop($$props, 'on_select_column', 3, null),
		on_select_row = prop($$props, 'on_select_row', 3, null),
		el = prop($$props, 'el', 15, null);

	function truncate_text(text, max_length = null, is_image = false) {
		if (is_image) return String(text);

		const str = String(text);

		if (!max_length || max_length <= 0) return str;
		if (str.length <= max_length) return str;

		return str.slice(0, max_length) + "...";
	}

	let should_truncate = user_derived(() => !$$props.edit && max_chars() !== null && max_chars() > 0);

	let display_content = user_derived(() => editable()
		? value()
		: display_value() !== null ? display_value() : value());

	let display_text = user_derived(() => get(should_truncate)
		? truncate_text(get(display_content), max_chars(), datatype() === "image")
		: get(display_content));

	function use_focus(node) {
		requestAnimationFrame(() => {
			node.focus();
		});

		return {};
	}

	function handle_blur(event) {
		$$props.onblur?.({ blur_event: event, coords: $$props.coords });
	}

	function handle_keydown(event) {
		$$props.onkeydown?.(event);
	}

	function commit_change(checked) {
		handle_blur({ target: { value: value() } });
	}

	user_effect(() => {
		if (!$$props.edit) {
			// Shim blur on removal for Safari and Firefox
			handle_blur({ target: { value: value() } });
		}
	});

	var fragment = root$b();
	var node_1 = first_child(fragment);

	{
		var consequent = ($$anchor) => {
			var textarea = root_1$7();

			remove_textarea_child(textarea);
			textarea.__mousedown = (e) => e.stopPropagation();
			textarea.__click = (e) => e.stopPropagation();
			textarea.__keydown = handle_keydown;

			let classes;

			bind_this(textarea, ($$value) => el($$value), () => el());
			effect(() => bind_value(textarea, value));
			action(textarea, ($$node) => use_focus?.($$node));

			template_effect(() => {
				textarea.readOnly = is_static();
				set_attribute(textarea, 'aria-readonly', is_static());
				set_attribute(textarea, 'aria-label', is_static() ? "Cell is read-only" : "Edit cell");
				classes = set_class(textarea, 1, 'svelte-odwpey', null, classes, { header: header() });
			});

			event('blur', textarea, handle_blur);
			append($$anchor, textarea);
		};

		if_block(node_1, ($$render) => {
			if ($$props.edit && datatype() !== "bool") $$render(consequent);
		});
	}

	var node_2 = sibling(node_1, 2);

	{
		var consequent_1 = ($$anchor) => {
			BooleanCell($$anchor, {
				get editable() {
					return editable();
				},
				on_change: commit_change,
				get value() {
					return value();
				},

				set value($$value) {
					value($$value);
				}
			});
		};

		var alternate_3 = ($$anchor) => {
			var span = root_3$4();

			span.__keydown = handle_keydown;

			let classes_1;
			var node_3 = child(span);

			{
				var consequent_2 = ($$anchor) => {
					const ImageComponent = user_derived(() => components().image);
					var fragment_2 = comment();
					var node_4 = first_child(fragment_2);

					{
						let $0 = user_derived(() => ({ url: get(display_text) }));

						component(node_4, () => get(ImageComponent), ($$anchor, ImageComponent_1) => {
							ImageComponent_1($$anchor, {
								get value() {
									return get($0);
								},
								show_label: false,
								label: 'cell-image',
								show_download_button: false,
								get i18n() {
									return $$props.i18n;
								},
								gradio: { dispatch: () => {} }
							});
						});
					}

					append($$anchor, fragment_2);
				};

				var alternate_2 = ($$anchor) => {
					var fragment_3 = comment();
					var node_5 = first_child(fragment_3);

					{
						var consequent_3 = ($$anchor) => {
							var fragment_4 = comment();
							var node_6 = first_child(fragment_4);

							html(node_6, () => get(display_text));
							append($$anchor, fragment_4);
						};

						var alternate_1 = ($$anchor) => {
							var fragment_5 = comment();
							var node_7 = first_child(fragment_5);

							{
								var consequent_4 = ($$anchor) => {
									{
										let $0 = user_derived(() => get(display_text).toLocaleString());

										MarkdownCode($$anchor, {
											get message() {
												return get($0);
											},

											get latex_delimiters() {
												return $$props.latex_delimiters;
											},

											get line_breaks() {
												return line_breaks();
											},
											chatbot: false
										});
									}
								};

								var alternate = ($$anchor) => {
									var text_1 = text$1();

									template_effect(() => set_text(text_1, get(display_text)));
									append($$anchor, text_1);
								};

								if_block(
									node_7,
									($$render) => {
										if (datatype() === "markdown") $$render(consequent_4); else $$render(alternate, false);
									},
									true
								);
							}

							append($$anchor, fragment_5);
						};

						if_block(
							node_5,
							($$render) => {
								if (datatype() === "html") $$render(consequent_3); else $$render(alternate_1, false);
							},
							true
						);
					}

					append($$anchor, fragment_3);
				};

				if_block(node_3, ($$render) => {
					if (datatype() === "image" && components().image) $$render(consequent_2); else $$render(alternate_2, false);
				});
			}

			reset(span);

			template_effect(() => {
				set_style(span, styling());
				set_attribute(span, 'data-editable', editable());
				set_attribute(span, 'data-max-chars', max_chars());
				set_attribute(span, 'data-expanded', $$props.edit);

				classes_1 = set_class(span, 1, 'svelte-odwpey', null, classes_1, {
					dragging: is_dragging(),
					edit: $$props.edit,
					expanded: $$props.edit,
					multiline: header(),
					text: datatype() === "str",
					wrap: wrap_text()
				});
			});

			event('focus', span, (e) => e.preventDefault());
			append($$anchor, span);
		};

		if_block(node_2, ($$render) => {
			if (datatype() === "bool" && typeof value() === "boolean") $$render(consequent_1); else $$render(alternate_3, false);
		});
	}

	var node_8 = sibling(node_2, 2);

	{
		var consequent_5 = ($$anchor) => {
			var fragment_8 = root_10$1();
			var node_9 = first_child(fragment_8);

			SelectionButtons(node_9, {
				position: 'column',
				get coords() {
					return $$props.coords;
				},
				on_click: () => on_select_column()($$props.coords[1])
			});

			var node_10 = sibling(node_9, 2);

			SelectionButtons(node_10, {
				position: 'row',
				get coords() {
					return $$props.coords;
				},
				on_click: () => on_select_row()($$props.coords[0])
			});

			append($$anchor, fragment_8);
		};

		if_block(node_8, ($$render) => {
			if (show_selection_buttons() && $$props.coords && on_select_column() && on_select_row()) $$render(consequent_5);
		});
	}

	append($$anchor, fragment);
	pop();
}

delegate(['mousedown', 'click', 'keydown']);

var root$a = from_html(`<button aria-label="Open cell menu" class="cell-menu-button svelte-j6wkfp" aria-haspopup="menu">&#8942;</button>`);

function CellMenuButton($$anchor, $$props) {
	push($$props, true);

	var button = root$a();

	button.__click = function (...$$args) {
		$$props.on_click?.apply(this, $$args);
	};

	button.__touchstart = (event) => {
		event.preventDefault();

		const touch = event.touches[0];

		const mouseEvent = new MouseEvent("click", {
			clientX: touch.clientX,
			clientY: touch.clientY,
			bubbles: true,
			cancelable: true,
			view: window
		});

		$$props.on_click(mouseEvent);
	};

	append($$anchor, button);
	pop();
}

delegate(['click', 'touchstart']);

var root$9 = from_html(`<div class="wrapper svelte-4dxgzr" aria-label="Static column"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg></div>`);

function Padlock($$anchor, $$props) {

	let size = prop($$props, 'size', 3, 16);
	var div = root$9();
	var svg = child(div);

	reset(div);

	template_effect(() => {
		set_attribute(svg, 'width', size());
		set_attribute(svg, 'height', size());
	});

	append($$anchor, div);
}

var root$8 = from_svg(`<svg viewBox="0 0 24 24" width="20" height="20"><path d="M5 5H19" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path><path d="M8 9H16" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path><path d="M11 13H13" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path></svg>`);

function FilterIcon($$anchor) {
	var svg = root$8();

	append($$anchor, svg);
}

var root$7 = from_svg(`<svg viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 8L8 4L12 8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M8 4V12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"></path></svg>`);

function SortButtonUp($$anchor, $$props) {

	let size = prop($$props, 'size', 3, 16);
	var svg = root$7();

	template_effect(() => {
		set_attribute(svg, 'width', size());
		set_attribute(svg, 'height', size());
	});

	append($$anchor, svg);
}

var root$6 = from_svg(`<svg viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 8L8 12L12 8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M8 12V4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"></path></svg>`);

function SortButtonDown($$anchor, $$props) {

	let size = prop($$props, 'size', 3, 16);
	var svg = root$6();

	template_effect(() => {
		set_attribute(svg, 'width', size());
		set_attribute(svg, 'height', size());
	});

	append($$anchor, svg);
}

var root_2$4 = from_html(`<span class="filter-indicator svelte-1d6xqpb" aria-label="Filtered"><!></span>`);
var root_6$2 = from_html(`<span class="sort-priority svelte-1d6xqpb"> </span>`);
var root_3$3 = from_html(`<span class="sort-indicator svelte-1d6xqpb"><!> <!></span>`);
var root_1$6 = from_html(`<span class="header-icons svelte-1d6xqpb"><!> <!> <!></span>`);
var root$5 = from_html(`<th><div class="cell-wrap svelte-1d6xqpb"><div class="header-content svelte-1d6xqpb"><!></div> <!> <!></div></th>`);

function HeaderCell($$anchor, $$props) {
	push($$props, true);

	let is_editing = prop($$props, 'is_editing', 3, false),
		is_selected = prop($$props, 'is_selected', 3, false),
		is_static = prop($$props, 'is_static', 3, false),
		sort_direction = prop($$props, 'sort_direction', 3, null),
		sort_priority = prop($$props, 'sort_priority', 3, null),
		multi_sort = prop($$props, 'multi_sort', 3, false),
		is_filtered = prop($$props, 'is_filtered', 3, false),
		show_menu_button = prop($$props, 'show_menu_button', 3, false),
		is_first_column = prop($$props, 'is_first_column', 3, false),
		line_breaks = prop($$props, 'line_breaks', 3, true),
		editable = prop($$props, 'editable', 3, true),
		max_chars = prop($$props, 'max_chars', 3, undefined);

	var th = root$5();
	let classes;

	th.__click = (e) => $$props.onclick(e, $$props.col_idx);

	th.__mousedown = (e) => {
		e.preventDefault();
		e.stopPropagation();
	};

	var div = child(th);
	var div_1 = child(div);
	var node = child(div_1);

	{
		let $0 = user_derived(() => [$$props.col_idx, 0]);

		EditableCell(node, {
			get value() {
				return $$props.value;
			},

			get latex_delimiters() {
				return $$props.latex_delimiters;
			},

			get line_breaks() {
				return line_breaks();
			},

			get edit() {
				return is_editing();
			},

			onkeydown: (event) => {
				if (["Enter", "Escape", "Tab"].includes(event.key)) {
					$$props.on_end_edit(event.key);
				}
			},
			header: true,
			get editable() {
				return editable();
			},

			get is_static() {
				return is_static();
			},

			get i18n() {
				return $$props.i18n;
			},

			get max_chars() {
				return max_chars();
			},

			get coords() {
				return get($0);
			}
		});
	}

	reset(div_1);

	var node_1 = sibling(div_1, 2);

	{
		var consequent_5 = ($$anchor) => {
			var span = root_1$6();
			var node_2 = child(span);

			{
				var consequent = ($$anchor) => {
					var span_1 = root_2$4();
					var node_3 = child(span_1);

					FilterIcon(node_3);
					reset(span_1);
					append($$anchor, span_1);
				};

				if_block(node_2, ($$render) => {
					if (is_filtered()) $$render(consequent);
				});
			}

			var node_4 = sibling(node_2, 2);

			{
				var consequent_3 = ($$anchor) => {
					var span_2 = root_3$3();
					var node_5 = child(span_2);

					{
						var consequent_1 = ($$anchor) => {
							SortButtonUp($$anchor, { size: 13 });
						};

						var alternate = ($$anchor) => {
							SortButtonDown($$anchor, { size: 13 });
						};

						if_block(node_5, ($$render) => {
							if (sort_direction() === "asc") $$render(consequent_1); else $$render(alternate, false);
						});
					}

					var node_6 = sibling(node_5, 2);

					{
						var consequent_2 = ($$anchor) => {
							var span_3 = root_6$2();
							var text = child(span_3, true);

							reset(span_3);
							template_effect(() => set_text(text, sort_priority()));
							append($$anchor, span_3);
						};

						if_block(node_6, ($$render) => {
							if (multi_sort() && sort_priority() != null) $$render(consequent_2);
						});
					}

					reset(span_2);
					template_effect(() => set_attribute(span_2, 'aria-label', `Sorted ${sort_direction() ?? ''}ending`));
					append($$anchor, span_2);
				};

				if_block(node_4, ($$render) => {
					if (sort_direction()) $$render(consequent_3);
				});
			}

			var node_7 = sibling(node_4, 2);

			{
				var consequent_4 = ($$anchor) => {
					Padlock($$anchor, { size: 11 });
				};

				if_block(node_7, ($$render) => {
					if (is_static()) $$render(consequent_4);
				});
			}

			reset(span);
			append($$anchor, span);
		};

		if_block(node_1, ($$render) => {
			if (sort_direction() || is_filtered() || is_static()) $$render(consequent_5);
		});
	}

	var node_8 = sibling(node_1, 2);

	{
		var consequent_6 = ($$anchor) => {
			CellMenuButton($$anchor, { on_click: (e) => $$props.on_menu_click(e, $$props.col_idx) });
		};

		if_block(node_8, ($$render) => {
			if (show_menu_button()) $$render(consequent_6);
		});
	}

	reset(div);
	reset(th);

	template_effect(() => {
		classes = set_class(th, 1, 'header-cell svelte-1d6xqpb', null, classes, {
			focus: is_editing() || is_selected(),
			sorted: sort_direction() !== null,
			filtered: is_filtered(),
			'first-column': is_first_column()
		});

		set_attribute(th, 'title', $$props.value);
	});

	append($$anchor, th);
	pop();
}

delegate(['click', 'mousedown']);

var root$4 = from_html(`<div><div class="cell-wrap svelte-183k2ki"><!> <!></div></div>`);

function DataCell($$anchor, $$props) {
	let display_value = prop($$props, 'display_value', 3, null),
		datatype = prop($$props, 'datatype', 3, "str"),
		col_style = prop($$props, 'col_style', 3, ""),
		cell_style = prop($$props, 'cell_style', 3, ""),
		selection_classes = prop($$props, 'selection_classes', 3, ""),
		is_editing = prop($$props, 'is_editing', 3, false),
		is_flash = prop($$props, 'is_flash', 3, false),
		is_static = prop($$props, 'is_static', 3, false),
		show_menu_button = prop($$props, 'show_menu_button', 3, false),
		show_selection_buttons = prop($$props, 'show_selection_buttons', 3, false),
		is_first_column = prop($$props, 'is_first_column', 3, false),
		line_breaks = prop($$props, 'line_breaks', 3, true),
		editable = prop($$props, 'editable', 3, true),
		max_chars = prop($$props, 'max_chars', 3, undefined),
		components = prop($$props, 'components', 19, () => ({})),
		is_dragging = prop($$props, 'is_dragging', 3, false),
		wrap_text = prop($$props, 'wrap_text', 3, false);
	var div = root$4();
	let classes;

	div.__mousedown = function (...$$args) {
		$$props.onmousedown?.apply(this, $$args);
	};

	div.__dblclick = function (...$$args) {
		$$props.ondblclick?.apply(this, $$args);
	};

	div.__contextmenu = function (...$$args) {
		$$props.oncontextmenu?.apply(this, $$args);
	};

	var div_1 = child(div);
	var node = child(div_1);

	{
		let $0 = user_derived(() => [$$props.row_idx, $$props.col_idx]);

		EditableCell(node, {
			get value() {
				return $$props.value;
			},

			get display_value() {
				return display_value();
			},

			get latex_delimiters() {
				return $$props.latex_delimiters;
			},

			get line_breaks() {
				return line_breaks();
			},

			get editable() {
				return editable();
			},

			get is_static() {
				return is_static();
			},

			get edit() {
				return is_editing();
			},

			get datatype() {
				return datatype();
			},

			get onblur() {
				return $$props.onblur;
			},

			get max_chars() {
				return max_chars();
			},

			get i18n() {
				return $$props.i18n;
			},

			get components() {
				return components();
			},

			get show_selection_buttons() {
				return show_selection_buttons();
			},

			get coords() {
				return get($0);
			},

			get on_select_column() {
				return $$props.on_select_column;
			},

			get on_select_row() {
				return $$props.on_select_row;
			},

			get is_dragging() {
				return is_dragging();
			},

			get wrap_text() {
				return wrap_text();
			}
		});
	}

	var node_1 = sibling(node, 2);

	{
		var consequent = ($$anchor) => {
			CellMenuButton($$anchor, {
				get on_click() {
					return $$props.on_menu_click;
				}
			});
		};

		if_block(node_1, ($$render) => {
			if (show_menu_button()) $$render(consequent);
		});
	}

	reset(div_1);
	reset(div);

	template_effect(() => {
		classes = set_class(div, 1, `body-cell ${selection_classes() ?? ''}`, 'svelte-183k2ki', classes, { flash: is_flash(), 'first-column': is_first_column() });
		set_attribute(div, 'data-row', $$props.row_idx);
		set_attribute(div, 'data-col', $$props.col_idx);
		set_attribute(div, 'data-testid', `cell-${$$props.row_idx}-${$$props.col_idx}`);
		set_style(div, `${col_style() ?? ''} ${cell_style() ?? ''}`);
	});

	append($$anchor, div);
}

delegate(['mousedown', 'dblclick', 'contextmenu']);

var root$3 = from_html(`<button class="add-row-button svelte-1at77md" aria-label="Add row">+</button>`);

function EmptyRowButton($$anchor, $$props) {

	var button = root$3();

	button.__click = function (...$$args) {
		$$props.on_click?.apply(this, $$args);
	};

	append($$anchor, button);
}

delegate(['click']);

function cast_value_to_type(v, t) {
  if (v === null || v === void 0) {
    return v;
  }
  if (t === "number") {
    const n = Number(v);
    return isNaN(n) ? v : n;
  }
  if (t === "bool") {
    if (typeof v === "boolean") return v;
    if (typeof v === "number") return v !== 0;
    const s = String(v).toLowerCase();
    if (s === "true" || s === "1") return true;
    if (s === "false" || s === "0") return false;
    return v;
  }
  if (t === "date") {
    const d = new Date(v);
    return isNaN(d.getTime()) ? v : d.toISOString();
  }
  return v;
}

var root_1$5 = from_svg(`<svg viewBox="0 0 24 24" width="16" height="16"><rect x="4" y="6" width="4" height="12" stroke="currentColor" stroke-width="2" fill="none"></rect><path d="M12 12H19M16 8L19 12L16 16" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"></path></svg>`);
var root_3$2 = from_svg(`<svg viewBox="0 0 24 24" width="16" height="16"><rect x="16" y="6" width="4" height="12" stroke="currentColor" stroke-width="2" fill="none"></rect><path d="M12 12H5M8 8L5 12L8 16" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"></path></svg>`);
var root_5$2 = from_svg(`<svg viewBox="0 0 24 24" width="16" height="16"><rect x="6" y="16" width="12" height="4" stroke="currentColor" stroke-width="2"></rect><path d="M12 12V5M8 8L12 5L16 8" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"></path></svg>`);
var root_7$1 = from_svg(`<svg viewBox="0 0 24 24" width="16" height="16"><rect x="6" y="4" width="12" height="4" stroke="currentColor" stroke-width="2"></rect><path d="M12 12V19M8 16L12 19L16 16" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"></path></svg>`);
var root_9$1 = from_svg(`<svg viewBox="0 0 24 24" width="16" height="16"><rect x="5" y="10" width="14" height="4" stroke="currentColor" stroke-width="2"></rect><path d="M8 7L16 17M16 7L8 17" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path></svg>`);
var root_11$1 = from_svg(`<svg viewBox="0 0 24 24" width="16" height="16"><rect x="10" y="5" width="4" height="14" stroke="currentColor" stroke-width="2"></rect><path d="M7 8L17 16M17 8L7 16" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path></svg>`);
var root_13 = from_svg(`<svg viewBox="0 0 24 24" width="16" height="16"><path d="M8 16L12 12L16 16" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"></path><path d="M12 12V19" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path><path d="M5 7H19" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path></svg>`);
var root_15$1 = from_svg(`<svg viewBox="0 0 24 24" width="16" height="16"><path d="M8 12L12 16L16 12" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"></path><path d="M12 16V9" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path><path d="M5 5H19" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path></svg>`);
var root_17 = from_svg(`<svg viewBox="0 0 24 24" width="16" height="16"><path d="M5 5H19" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path><path d="M5 9H15" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path><path d="M5 13H11" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path><path d="M5 17H7" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path><path d="M17 17L21 21M21 17L17 21" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path></svg>`);
var root_19 = from_svg(`<svg viewBox="0 0 24 24" width="16" height="16"><path d="M5 5H19" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path><path d="M8 9H16" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path><path d="M11 13H13" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path></svg>`);
var root_21 = from_svg(`<svg viewBox="0 0 24 24" width="16" height="16"><path d="M5 5H19" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path><path d="M8 9H16" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path><path d="M11 13H13" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path><path d="M17 17L21 21M21 17L17 21" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path></svg>`);

function CellMenuIcons($$anchor, $$props) {

	var fragment = comment();
	var node = first_child(fragment);

	{
		var consequent = ($$anchor) => {
			var svg = root_1$5();

			append($$anchor, svg);
		};

		var alternate_9 = ($$anchor) => {
			var fragment_1 = comment();
			var node_1 = first_child(fragment_1);

			{
				var consequent_1 = ($$anchor) => {
					var svg_1 = root_3$2();

					append($$anchor, svg_1);
				};

				var alternate_8 = ($$anchor) => {
					var fragment_2 = comment();
					var node_2 = first_child(fragment_2);

					{
						var consequent_2 = ($$anchor) => {
							var svg_2 = root_5$2();

							append($$anchor, svg_2);
						};

						var alternate_7 = ($$anchor) => {
							var fragment_3 = comment();
							var node_3 = first_child(fragment_3);

							{
								var consequent_3 = ($$anchor) => {
									var svg_3 = root_7$1();

									append($$anchor, svg_3);
								};

								var alternate_6 = ($$anchor) => {
									var fragment_4 = comment();
									var node_4 = first_child(fragment_4);

									{
										var consequent_4 = ($$anchor) => {
											var svg_4 = root_9$1();

											append($$anchor, svg_4);
										};

										var alternate_5 = ($$anchor) => {
											var fragment_5 = comment();
											var node_5 = first_child(fragment_5);

											{
												var consequent_5 = ($$anchor) => {
													var svg_5 = root_11$1();

													append($$anchor, svg_5);
												};

												var alternate_4 = ($$anchor) => {
													var fragment_6 = comment();
													var node_6 = first_child(fragment_6);

													{
														var consequent_6 = ($$anchor) => {
															var svg_6 = root_13();

															append($$anchor, svg_6);
														};

														var alternate_3 = ($$anchor) => {
															var fragment_7 = comment();
															var node_7 = first_child(fragment_7);

															{
																var consequent_7 = ($$anchor) => {
																	var svg_7 = root_15$1();

																	append($$anchor, svg_7);
																};

																var alternate_2 = ($$anchor) => {
																	var fragment_8 = comment();
																	var node_8 = first_child(fragment_8);

																	{
																		var consequent_8 = ($$anchor) => {
																			var svg_8 = root_17();

																			append($$anchor, svg_8);
																		};

																		var alternate_1 = ($$anchor) => {
																			var fragment_9 = comment();
																			var node_9 = first_child(fragment_9);

																			{
																				var consequent_9 = ($$anchor) => {
																					var svg_9 = root_19();

																					append($$anchor, svg_9);
																				};

																				var alternate = ($$anchor) => {
																					var fragment_10 = comment();
																					var node_10 = first_child(fragment_10);

																					{
																						var consequent_10 = ($$anchor) => {
																							var svg_10 = root_21();

																							append($$anchor, svg_10);
																						};

																						if_block(
																							node_10,
																							($$render) => {
																								if ($$props.icon == "clear-filter") $$render(consequent_10);
																							},
																							true
																						);
																					}

																					append($$anchor, fragment_10);
																				};

																				if_block(
																					node_9,
																					($$render) => {
																						if ($$props.icon == "filter") $$render(consequent_9); else $$render(alternate, false);
																					},
																					true
																				);
																			}

																			append($$anchor, fragment_9);
																		};

																		if_block(
																			node_8,
																			($$render) => {
																				if ($$props.icon == "clear-sort") $$render(consequent_8); else $$render(alternate_1, false);
																			},
																			true
																		);
																	}

																	append($$anchor, fragment_8);
																};

																if_block(
																	node_7,
																	($$render) => {
																		if ($$props.icon == "sort-desc") $$render(consequent_7); else $$render(alternate_2, false);
																	},
																	true
																);
															}

															append($$anchor, fragment_7);
														};

														if_block(
															node_6,
															($$render) => {
																if ($$props.icon == "sort-asc") $$render(consequent_6); else $$render(alternate_3, false);
															},
															true
														);
													}

													append($$anchor, fragment_6);
												};

												if_block(
													node_5,
													($$render) => {
														if ($$props.icon == "delete-column") $$render(consequent_5); else $$render(alternate_4, false);
													},
													true
												);
											}

											append($$anchor, fragment_5);
										};

										if_block(
											node_4,
											($$render) => {
												if ($$props.icon == "delete-row") $$render(consequent_4); else $$render(alternate_5, false);
											},
											true
										);
									}

									append($$anchor, fragment_4);
								};

								if_block(
									node_3,
									($$render) => {
										if ($$props.icon == "add-row-below") $$render(consequent_3); else $$render(alternate_6, false);
									},
									true
								);
							}

							append($$anchor, fragment_3);
						};

						if_block(
							node_2,
							($$render) => {
								if ($$props.icon == "add-row-above") $$render(consequent_2); else $$render(alternate_7, false);
							},
							true
						);
					}

					append($$anchor, fragment_2);
				};

				if_block(
					node_1,
					($$render) => {
						if ($$props.icon == "add-column-left") $$render(consequent_1); else $$render(alternate_8, false);
					},
					true
				);
			}

			append($$anchor, fragment_1);
		};

		if_block(node, ($$render) => {
			if ($$props.icon == "add-column-right") $$render(consequent); else $$render(alternate_9, false);
		});
	}

	append($$anchor, fragment);
}

var root_2$3 = from_html(`<button class="filter-option svelte-1y6dxc7"> </button>`);
var root_1$4 = from_html(`<div class="dropdown-filter-options svelte-1y6dxc7"></div>`);
var root$2 = from_html(`<div><div class="background svelte-1y6dxc7"></div> <div class="filter-menu svelte-1y6dxc7"><div class="filter-datatype-container svelte-1y6dxc7"><span class="svelte-1y6dxc7">Filter as</span> <button class="svelte-1y6dxc7"> </button></div> <div class="input-container svelte-1y6dxc7"><div class="filter-dropdown"><button class="svelte-1y6dxc7"> <!></button> <!></div> <input type="text" placeholder="Type a value" class="filter-input svelte-1y6dxc7"/></div> <button class="check-button svelte-1y6dxc7"><!></button></div></div>`);

function FilterMenu($$anchor, $$props) {
	push($$props, true);

	let on_filter = prop($$props, 'on_filter', 3, () => {});
	let menu_element;
	let datatype = state("string");
	let current_filter = state("Contains");
	let filter_dropdown_open = state(false);
	let filter_input_value = state("");

	const filter_options = {
		string: [
			"Contains",
			"Does not contain",
			"Starts with",
			"Ends with",
			"Is",
			"Is not",
			"Is empty",
			"Is not empty"
		],
		number: ["=", "≠", ">", "<", "≥", "≤", "Is empty", "Is not empty"]
	};

	onMount(() => {
		position_menu();
	});

	function position_menu() {
		if (!menu_element) return;

		const viewport_width = window.innerWidth;
		const viewport_height = window.innerHeight;
		const menu_rect = menu_element.getBoundingClientRect();
		const x = (viewport_width - menu_rect.width) / 2;
		const y = (viewport_height - menu_rect.height) / 2;

		menu_element.style.left = `${x}px`;
		menu_element.style.top = `${y}px`;
	}

	function handle_filter_input(e) {
		const target = e.target;

		set(filter_input_value, target.value, true);
	}

	var div = root$2();
	var div_1 = sibling(child(div), 2);
	var div_2 = child(div_1);
	var button = sibling(child(div_2), 2);

	button.__click = (e) => {
		e.stopPropagation();
		set(datatype, get(datatype) === "string" ? "number" : "string", true);
		set(current_filter, filter_options[get(datatype)][0], true);
	};

	var text = child(button, true);

	reset(button);
	reset(div_2);

	var div_3 = sibling(div_2, 2);
	var div_4 = child(div_3);
	var button_1 = child(div_4);

	button_1.__click = (e) => {
		e.stopPropagation();
		set(filter_dropdown_open, !get(filter_dropdown_open));
	};

	var text_1 = child(button_1);
	var node = sibling(text_1);

	DropdownArrow(node);
	reset(button_1);

	var node_1 = sibling(button_1, 2);

	{
		var consequent = ($$anchor) => {
			var div_5 = root_1$4();

			each(div_5, 21, () => filter_options[get(datatype)], index, ($$anchor, opt) => {
				var button_2 = root_2$3();

				button_2.__click = (e) => {
					e.stopPropagation();
					set(current_filter, get(opt), true);
					set(filter_dropdown_open, !get(filter_dropdown_open));
				};

				var text_2 = child(button_2, true);

				reset(button_2);
				template_effect(() => set_text(text_2, get(opt)));
				append($$anchor, button_2);
			});

			reset(div_5);
			append($$anchor, div_5);
		};

		if_block(node_1, ($$render) => {
			if (get(filter_dropdown_open)) $$render(consequent);
		});
	}

	reset(div_4);

	var input = sibling(div_4, 2);

	remove_input_defaults(input);
	input.__click = (e) => e.stopPropagation();
	input.__input = handle_filter_input;
	reset(div_3);

	var button_3 = sibling(div_3, 2);

	button_3.__click = () => on_filter()(get(datatype), get(current_filter), get(filter_input_value));

	var node_2 = child(button_3);

	Check(node_2);
	reset(button_3);
	reset(div_1);
	bind_this(div_1, ($$value) => menu_element = $$value, () => menu_element);
	reset(div);

	template_effect(() => {
		set_attribute(button, 'aria-label', `Change filter type. Filtering ${get(datatype)}s`);
		set_text(text, get(datatype));
		set_attribute(button_1, 'aria-label', `Change filter. Using '${get(current_filter)}'`);
		set_text(text_1, `${get(current_filter) ?? ''} `);
		set_value(input, get(filter_input_value));
	});

	append($$anchor, div);
	pop();
}

delegate(['click', 'input']);

var root_2$2 = from_html(`<span class="priority svelte-1v4jjjx"> </span>`);
var root_3$1 = from_html(`<span class="priority svelte-1v4jjjx"> </span>`);
var root_4$1 = from_html(`<span class="priority svelte-1v4jjjx">1</span>`);
var root_1$3 = from_html(`<button role="menuitem"><!> <!></button> <button role="menuitem"><!> <!></button> <button role="menuitem" class="svelte-1v4jjjx"><!> </button> <button role="menuitem"><!> <!></button> <button role="menuitem" class="svelte-1v4jjjx"><!> </button>`, 1);
var root_6$1 = from_html(`<button role="menuitem" class="delete svelte-1v4jjjx" aria-label="Delete row"><!> </button>`);
var root_5$1 = from_html(`<button role="menuitem" aria-label="Add row above" class="svelte-1v4jjjx"><!> </button> <button role="menuitem" aria-label="Add row below" class="svelte-1v4jjjx"><!> </button> <!>`, 1);
var root_8 = from_html(`<button role="menuitem" class="delete svelte-1v4jjjx" aria-label="Delete column"><!> </button>`);
var root_7 = from_html(`<button role="menuitem" aria-label="Add column to the left" class="svelte-1v4jjjx"><!> </button> <button role="menuitem" aria-label="Add column to the right" class="svelte-1v4jjjx"><!> </button> <!>`, 1);
var root$1 = from_html(`<div class="cell-menu svelte-1v4jjjx" role="menu"><!> <!> <!></div> <!>`, 1);

function CellMenu($$anchor, $$props) {
	push($$props, true);

	let on_sort = prop($$props, 'on_sort', 3, () => {}),
		on_clear_sort = prop($$props, 'on_clear_sort', 3, () => {}),
		sort_direction = prop($$props, 'sort_direction', 3, null),
		sort_priority = prop($$props, 'sort_priority', 3, null),
		on_filter = prop($$props, 'on_filter', 3, () => {}),
		on_clear_filter = prop($$props, 'on_clear_filter', 3, () => {}),
		filter_active = prop($$props, 'filter_active', 3, null),
		editable = prop($$props, 'editable', 3, true);

	let menu_element;
	let active_filter_menu = state(null);
	let is_header = user_derived(() => $$props.row === -1);
	let can_add_rows = user_derived(() => editable() && $$props.row_count[1] === "dynamic");
	let can_add_columns = user_derived(() => editable() && $$props.col_count[1] === "dynamic");

	onMount(() => {
		position_menu();
	});

	function position_menu() {
		if (!menu_element) return;

		const viewport_width = window.innerWidth;
		const viewport_height = window.innerHeight;
		const menu_rect = menu_element.getBoundingClientRect();
		let new_x = $$props.x - 30;
		let new_y = $$props.y - 20;

		if (new_x + menu_rect.width > viewport_width) {
			new_x = $$props.x - menu_rect.width + 10;
		}

		if (new_y + menu_rect.height > viewport_height) {
			new_y = $$props.y - menu_rect.height + 10;
		}

		menu_element.style.left = `${new_x}px`;
		menu_element.style.top = `${new_y}px`;
	}

	function toggle_filter_menu() {
		if (filter_active()) {
			on_filter()("string", "", "");

			return;
		}

		const menu_rect = menu_element.getBoundingClientRect();

		set(active_filter_menu, { x: menu_rect.right, y: menu_rect.top + menu_rect.height / 2 }, true);
	}

	var fragment = root$1();
	var div = first_child(fragment);
	var node = child(div);

	{
		var consequent_3 = ($$anchor) => {
			var fragment_1 = root_1$3();
			var button = first_child(fragment_1);

			button.__click = () => on_sort()("asc");

			let classes;
			var node_1 = child(button);

			CellMenuIcons(node_1, { icon: 'sort-asc' });

			var text = sibling(node_1);
			var node_2 = sibling(text);

			{
				var consequent = ($$anchor) => {
					var span = root_2$2();
					var text_1 = child(span, true);

					reset(span);
					template_effect(() => set_text(text_1, sort_priority()));
					append($$anchor, span);
				};

				if_block(node_2, ($$render) => {
					if (sort_direction() === "asc" && sort_priority() !== null) $$render(consequent);
				});
			}

			reset(button);

			var button_1 = sibling(button, 2);

			button_1.__click = () => on_sort()("desc");

			let classes_1;
			var node_3 = child(button_1);

			CellMenuIcons(node_3, { icon: 'sort-desc' });

			var text_2 = sibling(node_3);
			var node_4 = sibling(text_2);

			{
				var consequent_1 = ($$anchor) => {
					var span_1 = root_3$1();
					var text_3 = child(span_1, true);

					reset(span_1);
					template_effect(() => set_text(text_3, sort_priority()));
					append($$anchor, span_1);
				};

				if_block(node_4, ($$render) => {
					if (sort_direction() === "desc" && sort_priority() !== null) $$render(consequent_1);
				});
			}

			reset(button_1);

			var button_2 = sibling(button_1, 2);

			button_2.__click = function (...$$args) {
				on_clear_sort()?.apply(this, $$args);
			};

			var node_5 = child(button_2);

			CellMenuIcons(node_5, { icon: 'clear-sort' });

			var text_4 = sibling(node_5);

			reset(button_2);

			var button_3 = sibling(button_2, 2);

			button_3.__click = (e) => {
				e.stopPropagation();
				toggle_filter_menu();
			};

			let classes_2;
			var node_6 = child(button_3);

			CellMenuIcons(node_6, { icon: 'filter' });

			var text_5 = sibling(node_6);
			var node_7 = sibling(text_5);

			{
				var consequent_2 = ($$anchor) => {
					var span_2 = root_4$1();

					append($$anchor, span_2);
				};

				if_block(node_7, ($$render) => {
					if (filter_active()) $$render(consequent_2);
				});
			}

			reset(button_3);

			var button_4 = sibling(button_3, 2);

			button_4.__click = function (...$$args) {
				on_clear_filter()?.apply(this, $$args);
			};

			var node_8 = child(button_4);

			CellMenuIcons(node_8, { icon: 'clear-filter' });

			var text_6 = sibling(node_8);

			reset(button_4);

			template_effect(
				($0, $1, $2, $3, $4) => {
					classes = set_class(button, 1, 'svelte-1v4jjjx', null, classes, { active: sort_direction() === "asc" });
					set_text(text, ` ${$0 ?? ''} `);
					classes_1 = set_class(button_1, 1, 'svelte-1v4jjjx', null, classes_1, { active: sort_direction() === "desc" });
					set_text(text_2, ` ${$1 ?? ''} `);
					set_text(text_4, ` ${$2 ?? ''}`);
					classes_2 = set_class(button_3, 1, 'svelte-1v4jjjx', null, classes_2, { active: filter_active() || get(active_filter_menu) });
					set_text(text_5, ` ${$3 ?? ''} `);
					set_text(text_6, ` ${$4 ?? ''}`);
				},
				[
					() => $$props.i18n("dataframe.sort_ascending"),
					() => $$props.i18n("dataframe.sort_descending"),
					() => $$props.i18n("dataframe.clear_sort"),
					() => $$props.i18n("dataframe.filter"),
					() => $$props.i18n("dataframe.clear_filter")
				]
			);

			append($$anchor, fragment_1);
		};

		if_block(node, ($$render) => {
			if (get(is_header)) $$render(consequent_3);
		});
	}

	var node_9 = sibling(node, 2);

	{
		var consequent_5 = ($$anchor) => {
			var fragment_2 = root_5$1();
			var button_5 = first_child(fragment_2);

			button_5.__click = () => $$props.on_add_row_above();

			var node_10 = child(button_5);

			CellMenuIcons(node_10, { icon: 'add-row-above' });

			var text_7 = sibling(node_10);

			reset(button_5);

			var button_6 = sibling(button_5, 2);

			button_6.__click = () => $$props.on_add_row_below();

			var node_11 = child(button_6);

			CellMenuIcons(node_11, { icon: 'add-row-below' });

			var text_8 = sibling(node_11);

			reset(button_6);

			var node_12 = sibling(button_6, 2);

			{
				var consequent_4 = ($$anchor) => {
					var button_7 = root_6$1();

					button_7.__click = function (...$$args) {
						$$props.on_delete_row?.apply(this, $$args);
					};

					var node_13 = child(button_7);

					CellMenuIcons(node_13, { icon: 'delete-row' });

					var text_9 = sibling(node_13);

					reset(button_7);
					template_effect(($0) => set_text(text_9, ` ${$0 ?? ''}`), [() => $$props.i18n("dataframe.delete_row")]);
					append($$anchor, button_7);
				};

				if_block(node_12, ($$render) => {
					if ($$props.can_delete_rows) $$render(consequent_4);
				});
			}

			template_effect(
				($0, $1) => {
					set_text(text_7, ` ${$0 ?? ''}`);
					set_text(text_8, ` ${$1 ?? ''}`);
				},
				[
					() => $$props.i18n("dataframe.add_row_above"),
					() => $$props.i18n("dataframe.add_row_below")
				]
			);

			append($$anchor, fragment_2);
		};

		if_block(node_9, ($$render) => {
			if (!get(is_header) && get(can_add_rows)) $$render(consequent_5);
		});
	}

	var node_14 = sibling(node_9, 2);

	{
		var consequent_7 = ($$anchor) => {
			var fragment_3 = root_7();
			var button_8 = first_child(fragment_3);

			button_8.__click = () => $$props.on_add_column_left();

			var node_15 = child(button_8);

			CellMenuIcons(node_15, { icon: 'add-column-left' });

			var text_10 = sibling(node_15);

			reset(button_8);

			var button_9 = sibling(button_8, 2);

			button_9.__click = () => $$props.on_add_column_right();

			var node_16 = child(button_9);

			CellMenuIcons(node_16, { icon: 'add-column-right' });

			var text_11 = sibling(node_16);

			reset(button_9);

			var node_17 = sibling(button_9, 2);

			{
				var consequent_6 = ($$anchor) => {
					var button_10 = root_8();

					button_10.__click = function (...$$args) {
						$$props.on_delete_col?.apply(this, $$args);
					};

					var node_18 = child(button_10);

					CellMenuIcons(node_18, { icon: 'delete-column' });

					var text_12 = sibling(node_18);

					reset(button_10);
					template_effect(($0) => set_text(text_12, ` ${$0 ?? ''}`), [() => $$props.i18n("dataframe.delete_column")]);
					append($$anchor, button_10);
				};

				if_block(node_17, ($$render) => {
					if ($$props.can_delete_cols) $$render(consequent_6);
				});
			}

			template_effect(
				($0, $1) => {
					set_text(text_10, ` ${$0 ?? ''}`);
					set_text(text_11, ` ${$1 ?? ''}`);
				},
				[
					() => $$props.i18n("dataframe.add_column_left"),
					() => $$props.i18n("dataframe.add_column_right")
				]
			);

			append($$anchor, fragment_3);
		};

		if_block(node_14, ($$render) => {
			if (get(can_add_columns)) $$render(consequent_7);
		});
	}

	reset(div);
	bind_this(div, ($$value) => menu_element = $$value, () => menu_element);

	var node_19 = sibling(div, 2);

	{
		var consequent_8 = ($$anchor) => {
			FilterMenu($$anchor, {
				get on_filter() {
					return on_filter();
				}
			});
		};

		if_block(node_19, ($$render) => {
			if (get(active_filter_menu)) $$render(consequent_8);
		});
	}

	append($$anchor, fragment);
	pop();
}

delegate(['click']);

var root_2$1 = from_html(`<button class="toolbar-button check-button svelte-1rajnm3" aria-label="Apply filter and update dataframe values" title="Apply filter and update dataframe values"><!></button>`);
var root_1$2 = from_html(`<div class="search-container svelte-1rajnm3"><input type="text"/> <!></div>`);
var root = from_html(`<div class="toolbar svelte-1rajnm3" role="toolbar" aria-label="Table actions"><div class="toolbar-buttons svelte-1rajnm3"><!> <!> <!></div></div>`);

function Toolbar($$anchor, $$props) {
	push($$props, true);

	let show_fullscreen_button = prop($$props, 'show_fullscreen_button', 3, false),
		show_copy_button = prop($$props, 'show_copy_button', 3, false),
		show_search = prop($$props, 'show_search', 3, "none"),
		fullscreen = prop($$props, 'fullscreen', 3, false),
		current_search_query = prop($$props, 'current_search_query', 7, null);

	let copied = state(false);
	let timer;
	let input_value = state("");

	function handle_search_input(e) {
		const target = e.target;

		set(input_value, target.value, true);

		const new_query = get(input_value) || null;

		if (current_search_query() !== new_query) {
			current_search_query(new_query);
			$$props.onsearch?.(current_search_query());
		}
	}

	function copy_feedback() {
		set(copied, true);

		if (timer) clearTimeout(timer);

		timer = setTimeout(
			() => {
				set(copied, false);
			},
			2000
		);
	}

	async function handle_copy() {
		await $$props.on_copy();
		copy_feedback();
	}

	user_effect(() => {
		return () => {
			if (timer) clearTimeout(timer);
		};
	});

	var div = root();
	var div_1 = child(div);
	var node = child(div_1);

	{
		var consequent_1 = ($$anchor) => {
			var div_2 = root_1$2();
			var input = child(div_2);

			remove_input_defaults(input);
			input.__input = handle_search_input;

			let classes;
			var node_1 = sibling(input, 2);

			{
				var consequent = ($$anchor) => {
					var button = root_2$1();

					button.__click = function (...$$args) {
						$$props.on_commit_filter?.apply(this, $$args);
					};

					var node_2 = child(button);

					Check(node_2);
					reset(button);
					append($$anchor, button);
				};

				if_block(node_1, ($$render) => {
					if (current_search_query() && show_search() === "filter") $$render(consequent);
				});
			}

			reset(div_2);

			template_effect(() => {
				set_value(input, current_search_query() || "");
				set_attribute(input, 'placeholder', show_search() === "filter" ? "Filter..." : "Search...");
				classes = set_class(input, 1, 'search-input svelte-1rajnm3', null, classes, { 'filter-mode': show_search() === "filter" });
				set_attribute(input, 'title', `Enter text to ${show_search()} the table`);
			});

			append($$anchor, div_2);
		};

		if_block(node, ($$render) => {
			if (show_search() !== "none") $$render(consequent_1);
		});
	}

	var node_3 = sibling(node, 2);

	{
		var consequent_2 = ($$anchor) => {
			{
				let $0 = user_derived(() => get(copied) ? Check : Copy);
				let $1 = user_derived(() => get(copied) ? "Copied to clipboard" : "Copy table data");

				IconButton($$anchor, {
					get Icon() {
						return get($0);
					},

					get label() {
						return get($1);
					},
					onclick: handle_copy
				});
			}
		};

		if_block(node_3, ($$render) => {
			if (show_copy_button()) $$render(consequent_2);
		});
	}

	var node_4 = sibling(node_3, 2);

	{
		var consequent_3 = ($$anchor) => {
			FullscreenButton($$anchor, {
				get fullscreen() {
					return fullscreen();
				},
				onclick: (_fs) => $$props.onfullscreen?.()
			});
		};

		if_block(node_4, ($$render) => {
			if (show_fullscreen_button()) $$render(consequent_3);
		});
	}

	reset(div_1);
	reset(div);
	append($$anchor, div);
	pop();
}

delegate(['input', 'click']);

function is_cell_in_selection(coords, selected_cells) {
  const [row, col] = coords;
  return selected_cells.some(([r, c]) => r === row && c === col);
}
function is_cell_selected(cell, selected_cells) {
  const [row, col] = cell;
  if (!selected_cells.some(([r, c]) => r === row && c === col)) return "";
  const up = selected_cells.some(([r, c]) => r === row - 1 && c === col);
  const down = selected_cells.some(([r, c]) => r === row + 1 && c === col);
  const left = selected_cells.some(([r, c]) => r === row && c === col - 1);
  const right = selected_cells.some(([r, c]) => r === row && c === col + 1);
  return `cell-selected${up ? " no-top" : ""}${down ? " no-bottom" : ""}${left ? " no-left" : ""}${right ? " no-right" : ""}`;
}
function handle_click_outside(event, parent) {
  const [trigger] = event.composedPath();
  return !parent.contains(trigger);
}

async function copy_table_data(data, selected_cells) {
  if (!data || !data.length) return;
  const cells_to_copy = selected_cells || data.flatMap((row, r) => row.map((_, c) => [r, c]));
  const csv = cells_to_copy.reduce(
    (acc, [row, col]) => {
      acc[row] = acc[row] || {};
      const value = String(data[row][col].value);
      acc[row][col] = value.includes(",") || value.includes('"') || value.includes("\n") ? `"${value.replace(/"/g, '""')}"` : value;
      return acc;
    },
    {}
  );
  const rows = Object.keys(csv).sort((a, b) => +a - +b);
  if (!rows.length) return;
  const cols = Object.keys(csv[rows[0]]).sort((a, b) => +a - +b);
  const text = rows.map((r) => cols.map((c) => csv[r][c] || "").join(",")).join("\n");
  try {
    await navigator.clipboard.writeText(text);
  } catch (err) {
    throw new Error("Failed to copy to clipboard: " + err.message);
  }
}
function guess_delimiter(text, possibleDelimiters) {
  return possibleDelimiters.filter(weedOut);
  function weedOut(delimiter) {
    var cache = -1;
    return text.split("\n").every(checkLength);
    function checkLength(line) {
      if (!line) return true;
      var length = line.split(delimiter).length;
      if (cache < 0) cache = length;
      return cache === length && length > 1;
    }
  }
}
function data_uri_to_blob(data_uri) {
  const byte_str = atob(data_uri.split(",")[1]);
  const mime_str = data_uri.split(",")[0].split(":")[1].split(";")[0];
  const ab = new ArrayBuffer(byte_str.length);
  const ia = new Uint8Array(ab);
  for (let i = 0; i < byte_str.length; i++) {
    ia[i] = byte_str.charCodeAt(i);
  }
  return new Blob([ab], { type: mime_str });
}
function handle_file_upload(data_uri, update_headers, update_values) {
  const blob = data_uri_to_blob(data_uri);
  const reader = new FileReader();
  reader.addEventListener("loadend", (e) => {
    if (!e?.target?.result || typeof e.target.result !== "string") return;
    const [delimiter] = guess_delimiter(e.target.result, [",", "	"]);
    const [head, ...rest] = dsvFormat(delimiter).parseRows(e.target.result);
    update_headers(head);
    update_values(rest);
  });
  reader.readAsText(blob);
}

function gradio_filter_fn(row, columnId, filterValue) {
  const { dtype, filter, value: fval } = filterValue;
  const cell_value = String(row.getValue(columnId) ?? "");
  const compare_val = fval ?? "";
  if (dtype === "number") {
    const num = parseFloat(cell_value);
    const target = parseFloat(compare_val);
    if (isNaN(num) || isNaN(target)) {
      if (filter === "Is empty") return cell_value.trim() === "";
      if (filter === "Is not empty") return cell_value.trim() !== "";
      return true;
    }
    switch (filter) {
      case "=":
        return num === target;
      case "≠":
        return num !== target;
      case ">":
        return num > target;
      case "<":
        return num < target;
      case "≥":
        return num >= target;
      case "≤":
        return num <= target;
      case "Is empty":
        return cell_value.trim() === "";
      case "Is not empty":
        return cell_value.trim() !== "";
      default:
        return true;
    }
  }
  const lower = cell_value.toLowerCase();
  const target_lower = compare_val.toLowerCase();
  switch (filter) {
    case "Contains":
      return lower.includes(target_lower);
    case "Does not contain":
      return !lower.includes(target_lower);
    case "Starts with":
      return lower.startsWith(target_lower);
    case "Ends with":
      return lower.endsWith(target_lower);
    case "Is":
      return lower === target_lower;
    case "Is not":
      return lower !== target_lower;
    case "Is empty":
      return cell_value.trim() === "";
    case "Is not empty":
      return cell_value.trim() !== "";
    default:
      return true;
  }
}

/* column_measurement.svelte.ts generated by Svelte v5.48.0 */

function create_column_measurement(opts) {
	let col_widths = state(proxy([]));
	let total_header_width = state(0);
	let header_height = state(0);
	let col_lefts = state(proxy([]));

	function measure() {
		const current_row_el = opts.header_row_el();
		const current_table_el = opts.header_table_el();

		if (!current_row_el) return;

		const cells = current_row_el.querySelectorAll(".header-cell");
		const table_rect = current_table_el?.getBoundingClientRect();
		const table_left = table_rect?.left ?? 0;

		set(col_lefts, Array.from(cells).map((c) => c.getBoundingClientRect().left - table_left), true);
		set(col_widths, Array.from(cells).map((c) => c.getBoundingClientRect().width), true);

		if (current_table_el) {
			set(total_header_width, table_rect?.width ?? 0, true);
			set(header_height, table_rect?.height ?? 0, true);
		}

		opts.on_resize?.();
	}

	user_effect(() => {
		const table_el = opts.header_table_el();
		const row_el = opts.header_row_el();

		if (!table_el || !row_el) return;

		opts.resolved_headers();

		const ro = new ResizeObserver(measure);

		ro.observe(table_el);

		const cells = row_el.querySelectorAll(".header-cell");

		cells.forEach((cell) => ro.observe(cell));

		return () => ro.disconnect();
	});

	let row_num_width = user_derived(() => {
		if (!opts.show_row_numbers()) return 0;

		const row_el = opts.header_row_el();

		if (!row_el) return 0;

		const el = row_el.querySelector(".row-number-header");

		return el?.getBoundingClientRect().width ?? 48;
	});

	function get_col_style(index) {
		if (get(col_widths)[index] !== void 0) {
			return `width: ${get(col_widths)[index]}px; flex: 0 0 ${get(col_widths)[index]}px;`;
		}

		const user_widths = opts.column_widths();

		if (user_widths[index]) return `width: ${user_widths[index]};`;

		return "width: 150px;";
	}

	return {
		get col_widths() {
			return get(col_widths);
		},

		get col_lefts() {
			return get(col_lefts);
		},

		get total_header_width() {
			return get(total_header_width);
		},

		get header_height() {
			return get(header_height);
		},

		get row_num_width() {
			return get(row_num_width);
		},
		get_col_style
	};
}

var root_3 = from_html(`<div class="label svelte-2balj6"><p class="svelte-2balj6"> </p></div>`);
var root_2 = from_html(`<div class="header-row svelte-2balj6"><!> <!></div>`);
var root_5 = from_html(`<span class="sr-only svelte-2balj6"> </span>`);
var root_6 = from_html(`<th class="row-number-header svelte-2balj6">&nbsp;</th>`);
var root_10 = from_html(`<td class="row-number-cell svelte-2balj6"> </td>`);
var root_11 = from_html(`<td class="svelte-2balj6"><div class="cell-wrap svelte-2balj6"><!></div></td>`);
var root_9 = from_html(`<tr class="svelte-2balj6"><!><!></tr>`);
var root_16 = from_html(`<div class="row-number-cell svelte-2balj6" data-col="row-number"> </div>`);
var root_15 = from_html(`<div><!> <!></div>`);
var root_4 = from_html(`<div role="grid"><!> <table class="header-table svelte-2balj6"><thead class="svelte-2balj6"><tr class="svelte-2balj6"><!><!></tr></thead><tbody class="sizing-body svelte-2balj6" aria-hidden="true"><!></tbody></table> <div class="virtual-body svelte-2balj6"></div></div>`);
var root_18 = from_html(`<button class="scroll-top-button svelte-2balj6">&uarr;</button>`);
var root_1$1 = from_html(`<div><!> <div role="grid" tabindex="0"><!> <!></div></div> <!> <!>`, 1);

function Table($$anchor, $$props) {
	push($$props, true);

	let label = prop($$props, 'label', 3, null),
		show_label = prop($$props, 'show_label', 3, true),
		headers = prop($$props, 'headers', 31, () => proxy([])),
		values = prop($$props, 'values', 31, () => proxy([])),
		components = prop($$props, 'components', 19, () => ({})),
		editable = prop($$props, 'editable', 3, true),
		wrap = prop($$props, 'wrap', 3, false),
		max_height = prop($$props, 'max_height', 3, 500),
		line_breaks = prop($$props, 'line_breaks', 3, true),
		column_widths = prop($$props, 'column_widths', 19, () => []),
		show_row_numbers = prop($$props, 'show_row_numbers', 3, false),
		buttons = prop($$props, 'buttons', 3, null),
		value_is_output = prop($$props, 'value_is_output', 15, false),
		max_chars = prop($$props, 'max_chars', 3, undefined),
		show_search = prop($$props, 'show_search', 3, "none"),
		pinned_columns = prop($$props, 'pinned_columns', 3, 0),
		static_columns = prop($$props, 'static_columns', 19, () => []),
		fullscreen = prop($$props, 'fullscreen', 3, false),
		display_value = prop($$props, 'display_value', 3, null),
		styling = prop($$props, 'styling', 3, null);

	// convert values[][] into tanstack row objects
	let row_data = user_derived(() => (values() ?? []).map((row, i) => {
		const obj = { _index: i };

		(row ?? []).forEach((val, j) => {
			const dtype = Array.isArray($$props.datatype) ? $$props.datatype[j] : $$props.datatype;

			obj[`col_${j}`] = cast_value_to_type(val, dtype);
		});

		return obj;
	}));

	let resolved_headers = user_derived(() => {
		let h = headers() ?? [];

		if ($$props.col_count[1] === "fixed" && h.length < $$props.col_count[0]) {
			h = [
				...h,
				...Array($$props.col_count[0] - h.length).fill(null).map((_, i) => `${i + h.length}`)
			];
		}

		if (!h.length) {
			h = Array($$props.col_count[0]).fill(null).map((_, i) => `${i}`);
		}

		return h.map((v) => v ?? "");
	});

	let column_defs = user_derived(() => get(resolved_headers).map((header_value, j) => ({
		id: `col_${j}`,
		accessorKey: `col_${j}`,
		header: header_value,
		size: column_widths()[j] ? parseInt(column_widths()[j]) || 150 : 150,
		minSize: 45,
		filterFn: gradio_filter_fn,
		meta: {
			colIndex: j,
			datatype: Array.isArray($$props.datatype) ? $$props.datatype[j] : $$props.datatype,
			isStatic: static_columns().includes(j) || static_columns().includes(header_value),
			isPinned: j < pinned_columns()
		}
	})));

	let sorting = state(proxy([]));
	let column_filters = state(proxy([]));
	let global_filter = state("");

	let column_pinning = user_derived(() => ({
		left: get(column_defs).filter((_, i) => i < pinned_columns()).map((c) => c.id)
	}));

	const table = createSvelteTable({
		get data() {
			return get(row_data);
		},

		get columns() {
			return get(column_defs);
		},

		state: {
			get sorting() {
				return get(sorting);
			},

			get columnFilters() {
				return get(column_filters);
			},

			get globalFilter() {
				return get(global_filter);
			},

			get columnPinning() {
				return get(column_pinning);
			}
		},

		onSortingChange: (updater) => {
			set(sorting, typeof updater === "function" ? updater(get(sorting)) : updater, true);
		},

		onColumnFiltersChange: (updater) => {
			set(column_filters, typeof updater === "function" ? updater(get(column_filters)) : updater, true);
		},

		onGlobalFilterChange: (updater) => {
			set(global_filter, typeof updater === "function" ? updater(get(global_filter)) : updater, true);
		},
		getCoreRowModel: getCoreRowModel(),
		getSortedRowModel: getSortedRowModel(),
		getFilteredRowModel: getFilteredRowModel(),
		globalFilterFn: "includesString",
		enableSorting: true,
		enableMultiSort: true,
		maxMultiSortColCount: 3
	});

	let rows = user_derived(() => table.getRowModel().rows);
	let header_groups = user_derived(() => table.getHeaderGroups());
	let scroll_container;

	const virtualizer = createSvelteVirtualizer({
		get count() {
			return get(rows).length;
		},
		getScrollElement: () => scroll_container,
		estimateSize: () => 35,
		overscan: 10,
		measureElement: (el, _entry, instance) => {
			const h = el.getBoundingClientRect().height;

			if (h > 0) return h;

			const idx = el.getAttribute("data-index");

			if (idx != null) {
				const cached = instance.itemSizeCache?.get(Number(idx));

				if (typeof cached === "number") return cached;
			}

			return 35;
		}
	});

	let virtual_items = user_derived(() => virtualizer.virtualItems());
	let total_size = user_derived(() => virtualizer.totalSize());
	let selected_cells = state(proxy([]));
	let selected = state(false);
	let editing = state(false);
	let header_edit = state(false);
	let selected_header = state(false);
	let active_cell_menu = state(null);
	let active_header_menu = state(null);
	let copy_flash = state(false);
	let is_dragging = false;
	let show_scroll_button = state(false);
	let dragging = state(false // file drag
	);
	let parent;

	function get_dtype(col) {
		return Array.isArray($$props.datatype) ? $$props.datatype[col] ?? "str" : $$props.datatype;
	}

	function get_display_value(row, col) {
		if (display_value()?.[row]?.[col] !== undefined) return display_value()[row][col];

		return String(values()?.[row]?.[col] ?? "");
	}

	function get_styling(row, col) {
		return styling()?.[row]?.[col] ?? "";
	}

	function push_change(new_values, new_headers) {
		$$props.onchange?.({
			data: new_values ?? values(),
			headers: new_headers ?? get(resolved_headers),
			metadata: null
		});

		if (!value_is_output()) $$props.oninput?.();

		value_is_output(false);
	}

	function handle_cell_click(event, row, col) {
		event.preventDefault();
		event.stopPropagation();

		const coord = [row, col];

		if (event.shiftKey && get(selected)) {
			// range select
			const [r1, c1] = get(selected);

			const [r2, c2] = coord;
			const new_cells = [];

			for (let r = Math.min(r1, r2); r <= Math.max(r1, r2); r++) {
				for (let c = Math.min(c1, c2); c <= Math.max(c1, c2); c++) {
					new_cells.push([r, c]);
				}
			}

			set(selected_cells, new_cells, true);
		} else if (event.metaKey || event.ctrlKey) {
			// toggle select
			const exists = get(selected_cells).some(([r, c]) => r === row && c === col);

			set(
				selected_cells,
				exists
					? get(selected_cells).filter(([r, c]) => !(r === row && c === col))
					: [...get(selected_cells), coord],
				true
			);
		} else {
			set(selected_cells, [coord], true);
		}

		set(selected, coord, true);
		set(header_edit, false);
		set(selected_header, false);
		set(active_cell_menu, null);
		set(active_header_menu, null);

		// click selects, does NOT enter edit mode (double-click or typing does)
		set(editing, false);

		$$props.onselect?.({
			index: coord,
			value: values()?.[row]?.[col],
			row_value: values()?.[row] ?? [],
			col_value: values()?.map((r) => r[col]) ?? []
		});

		tick().then(() => parent?.focus());
	}

	function handle_cell_dblclick(event, row, col) {
		event.preventDefault();
		event.stopPropagation();

		if (!editable()) return;

		const col_is_static = static_columns().includes(col) || static_columns().includes(get(resolved_headers)[col]);

		if (!col_is_static) {
			set(editing, [row, col], true);
		}
	}

	function handle_blur(detail) {
		const { coords } = detail;
		const input_el = detail.blur_event.target;

		if (!input_el || input_el.value === undefined) return;

		const [row, col] = coords;
		const old_value = values()?.[row]?.[col];
		const new_value = input_el.value;

		if (String(old_value) !== String(new_value)) {
			const new_values = values().map((r) => [...r]);

			new_values[row][col] = new_value;
			values(new_values);

			$$props.onedit?.({
				index: [row, col],
				value: new_value,
				previous_value: String(old_value ?? "")
			});

			push_change(new_values);
		}
	}

	function handle_header_click(event, col) {
		if (event.target instanceof HTMLAnchorElement) return;

		event.preventDefault();
		event.stopPropagation();

		if (!editable()) return;

		set(editing, false);
		set(selected, false);
		set(selected_cells, [], true);
		set(active_cell_menu, null);
		set(active_header_menu, null);
		set(selected_header, col, true);
		set(header_edit, editable() ? col : false, true);
		parent?.focus();
	}

	function end_header_edit(key) {
		if (["Escape", "Enter", "Tab"].includes(key)) {
			set(header_edit, false);
			parent?.focus();
		}
	}

	function toggle_header_menu(event, col) {
		event.stopPropagation();

		if (get(active_header_menu)?.col === col) {
			set(active_header_menu, null);
		} else {
			const th = event.target.closest("th");

			if (th) {
				const rect = th.getBoundingClientRect();

				set(active_header_menu, { col, x: rect.right, y: rect.bottom }, true);
			}
		}
	}

	function handle_sort(col, direction) {
		const col_id = `col_${col}`;
		const desc = direction === "desc";

		// if already sorted this way, remove it
		const existing = get(sorting).findIndex((s) => s.id === col_id);

		if (existing >= 0 && get(sorting)[existing].desc === desc) {
			set(sorting, get(sorting).filter((s) => s.id !== col_id), true);
		} else {
			set(
				sorting,
				[
					...get(sorting).filter((s) => s.id !== col_id),
					{ id: col_id, desc }
				].slice(-3),
				true
			);
		}
	}

	function clear_sort() {
		set(sorting, [], true);
	}

	function handle_filter(col, dtype, filter, fvalue) {
		const col_id = `col_${col}`;
		const existing = get(column_filters).findIndex((f) => f.id === col_id);

		if (existing >= 0) {
			set(column_filters, get(column_filters).filter((f) => f.id !== col_id), true);
		} else {
			set(
				column_filters,
				[
					...get(column_filters),
					{ id: col_id, value: { dtype, filter, value: fvalue } }
				],
				true
			);
		}
	}

	function clear_filter() {
		set(column_filters, [], true);
	}

	function handle_search(query) {
		set(global_filter, query ?? "", true);
		$$props.onsearch?.(query);
	}

	function add_row(index) {
		if ($$props.row_count[1] !== "dynamic") return;

		const col_len = values()[0]?.length || get(resolved_headers).length || 1;
		const new_row = Array(col_len).fill("");
		const new_values = [...values()];

		if (index !== undefined) {
			new_values.splice(index, 0, new_row);
		} else {
			new_values.push(new_row);
		}

		values(new_values);
		push_change(new_values);
		set(selected, [index ?? new_values.length - 1, 0], true);
		parent?.focus();
	}

	function add_col(index) {
		if ($$props.col_count[1] !== "dynamic") return;

		const new_headers = [...headers() ?? [], `Header ${(headers()?.length ?? 0) + 1}`];
		const new_values = values().map((row) => [...row, ""]);

		if (index !== undefined) {
			new_headers.splice(index, 0, new_headers.pop());
			new_values.forEach((row) => row.splice(index, 0, row.pop()));
		}

		values(new_values);
		headers(new_headers);
		push_change(new_values, new_headers);
		parent?.focus();
	}

	function delete_row_at(index) {
		if (values().length <= 1) return;

		values([...values().slice(0, index), ...values().slice(index + 1)]);
		push_change(values());
		set(active_cell_menu, null);
		set(active_header_menu, null);
	}

	function delete_col_at(index) {
		if ($$props.col_count[1] !== "dynamic") return;
		if ((values()[0]?.length ?? 0) <= 1) return;

		values(values().map((row) => [...row.slice(0, index), ...row.slice(index + 1)]));

		headers([
			...(headers() ?? []).slice(0, index),
			...(headers() ?? []).slice(index + 1)
		]);

		push_change(values(), headers());
		set(active_cell_menu, null);
		set(active_header_menu, null);
		set(selected, false);
		set(selected_cells, [], true);
		set(editing, false);
	}

	function add_row_at(index, position) {
		add_row(position === "above" ? index : index + 1);
		set(active_cell_menu, null);
	}

	function add_col_at(index, position) {
		add_col(position === "left" ? index : index + 1);
		set(active_cell_menu, null);
	}

	// function handle_select_all(col: number, checked: boolean): void {
	// 	values = values.map((row) => {
	// 		const new_row = [...row];
	// 		new_row[col] = checked;
	// 		return new_row;
	// 	});
	// 	push_change(values);
	// }
	function commit_filter() {
		if (!get(global_filter) || show_search() !== "filter") return;

		// get the filtered rows from tanstack and push as new values
		const filtered_values = get(rows).map((row) => {
			const original_idx = row.original._index;

			return values()[original_idx];
		});

		values(filtered_values);
		set(global_filter, "");
		push_change(filtered_values);
	}

	async function handle_copy() {
		const data_for_copy = values().map((row) => row.map((val, j) => ({ id: `${j}`, value: val })));
		const cells_to_copy = get(selected_cells).length > 0 ? get(selected_cells) : null;

		await copy_table_data(data_for_copy, cells_to_copy);
		set(copy_flash, true);
		setTimeout(() => set(copy_flash, false), 800);
	}

	function handle_click_outside$1(event) {
		if (handle_click_outside(event, parent)) {
			set(selected_cells, [], true);
			set(selected, false);
			set(editing, false);
			set(header_edit, false);
			set(selected_header, false);
			set(active_cell_menu, null);
			set(active_header_menu, null);
		}
	}

	function handle_keydown(e) {
		if (!get(selected) && get(selected_header) === false) return;

		const num_rows = get(rows).length;
		const num_cols = get(resolved_headers).length;

		if (get(selected)) {
			const [row, col] = get(selected);

			switch (e.key) {
				case "ArrowUp":
					e.preventDefault();
					if (row > 0) {
						set(selected, [row - 1, col], true);
						set(selected_cells, [get(selected)], true);
						set(editing, false);
						virtualizer.instance.scrollToIndex(row - 1, { align: "auto" });
					}
					break;

				case "ArrowDown":
					e.preventDefault();
					if (row < num_rows - 1) {
						set(selected, [row + 1, col], true);
						set(selected_cells, [get(selected)], true);
						set(editing, false);
						virtualizer.instance.scrollToIndex(row + 1, { align: "auto" });
					}
					break;

				case "ArrowLeft":
					e.preventDefault();
					if (col > 0) {
						set(selected, [row, col - 1], true);
						set(selected_cells, [get(selected)], true);
						set(editing, false);
					}
					break;

				case "ArrowRight":
					e.preventDefault();
					if (col < num_cols - 1) {
						set(selected, [row, col + 1], true);
						set(selected_cells, [get(selected)], true);
						set(editing, false);
					}
					break;

				case "Tab":
					{
						e.preventDefault();

						const was_editing = !!get(editing);

						if (e.shiftKey) {
							if (col > 0) set(selected, [row, col - 1], true); else if (row > 0) set(selected, [row - 1, num_cols - 1], true);
						} else {
							if (col < num_cols - 1) set(selected, [row, col + 1], true); else if (row < num_rows - 1) set(selected, [row + 1, 0], true);
						}

						set(selected_cells, [get(selected)], true);

						if (was_editing) {
							const tab_col = get(selected)[1];
							const tab_static = static_columns().includes(tab_col) || static_columns().includes(get(resolved_headers)[tab_col]);

							set(editing, editable() && !tab_static ? get(selected) : false, true);
						} else {
							set(editing, false);
						}

						if (!get(editing)) tick().then(() => parent?.focus());

						break;
					}

				case "Enter":
					if (get(editing) && e.shiftKey) {
						// shift+enter inserts newline in textarea — don't intercept
						return;
					}
					e.preventDefault();
					if (get(editing)) {
						set(editing, false);

						if (row < num_rows - 1) {
							set(selected, [row + 1, col], true);
							set(selected_cells, [get(selected)], true);
						}

						tick().then(() => parent?.focus());
					} else if (editable()) {
						const enter_static = static_columns().includes(col) || static_columns().includes(get(resolved_headers)[col]);

						if (!enter_static) {
							set(editing, [row, col], true);
						}
					}
					break;

				case "Escape":
					set(editing, false);
					tick().then(() => parent?.focus());
					break;

				case "Delete":

				case "Backspace":
					if (!get(editing) && editable()) {
						e.preventDefault();

						const new_values = values().map((r) => [...r]);

						get(selected_cells).forEach(([r, c]) => {
							if (!static_columns().includes(c)) {
								new_values[r][c] = "";
							}
						});

						values(new_values);
						push_change(new_values);
					}
					break;

				default:
					// start editing on printable character
					if (editable() && !get(editing) && e.key.length === 1 && !e.ctrlKey && !e.metaKey && !static_columns().includes(col)) {
						set(editing, [row, col], true);
					}
					break;
			}

			if ((e.ctrlKey || e.metaKey) && e.key === "c") {
				handle_copy();
			}
		}
	}

	function handle_scroll() {
		if (scroll_container) {
			set(show_scroll_button, scroll_container.scrollTop > 100);
		}
	}

	function scroll_to_top() {
		scroll_container?.scrollTo({ top: 0 });
	}

	function toggle_cell_menu(event, row, col) {
		event.stopPropagation();

		if (get(active_cell_menu)?.row === row && get(active_cell_menu).col === col) {
			set(active_cell_menu, null);
		} else {
			const cell = event.target.closest(".body-cell, td");

			if (cell) {
				const rect = cell.getBoundingClientRect();

				set(active_cell_menu, { row, col, x: rect.right, y: rect.bottom }, true);
			}
		}
	}

	function on_file_upload(file_data) {
		handle_file_upload(
			typeof file_data === "string" ? file_data : file_data?.data ?? "",
			(head) => {
				headers(head.map((h) => h ?? ""));

				return headers().map((h, i) => ({ id: `h_${i}`, value: h }));
			},
			(vals) => {
				values(vals);
				push_change(vals, headers());
			}
		);
	}

	onMount(() => {
		document.addEventListener("click", handle_click_outside$1);

		return () => document.removeEventListener("click", handle_click_outside$1);
	});

	function measure_row(node) {
		tick().then(() => {
			console.log("measuring");
			virtualizer.instance.measureElement(node);
		});

		return { destroy() {} };
	}

	let header_row_el;
	let header_table_el;

	const measurement = create_column_measurement({
		header_row_el: () => header_row_el,
		header_table_el: () => header_table_el,
		resolved_headers: () => get(resolved_headers),
		row_data: () => get(row_data),
		show_row_numbers: () => show_row_numbers(),
		column_widths: () => column_widths(),
		on_resize: undefined
	});

	let disable_scroll = user_derived(() => get(active_cell_menu) !== null || get(active_header_menu) !== null);
	let selected_index = user_derived(() => get(selected) !== false ? get(selected)[0] : false);

	user_effect(() => {
		if (typeof get(selected_index) === "number") {
			virtualizer.instance.scrollToIndex(get(selected_index), { align: "auto" });
		}
	});

	function get_sort_info(col) {
		const col_id = `col_${col}`;
		const idx = get(sorting).findIndex((s) => s.id === col_id);

		if (idx === -1) return { direction: null, priority: null };

		return {
			direction: get(sorting)[idx].desc ? "desc" : "asc",
			priority: idx + 1
		};
	}

	function get_filter_active(col) {
		return get(column_filters).some((f) => f.id === `col_${col}`);
	}

	var fragment = root_1$1();

	event('resize', $window, () => {
		set(active_cell_menu, null);
		set(active_header_menu, null);
	});

	var div = first_child(fragment);
	let classes;
	var node_1 = child(div);

	{
		var consequent_1 = ($$anchor) => {
			var div_1 = root_2();
			var node_2 = child(div_1);

			{
				var consequent = ($$anchor) => {
					var div_2 = root_3();
					var p = child(div_2);
					var text = child(p, true);

					reset(p);
					reset(div_2);
					template_effect(() => set_text(text, label()));
					append($$anchor, div_2);
				};

				if_block(node_2, ($$render) => {
					if (label() && label().length !== 0 && show_label()) $$render(consequent);
				});
			}

			var node_3 = sibling(node_2, 2);

			{
				let $0 = user_derived(() => buttons() === null ? true : buttons().includes("fullscreen"));
				let $1 = user_derived(() => buttons() === null ? true : buttons().includes("copy"));
				let $2 = user_derived(() => get(global_filter) || null);

				Toolbar(node_3, {
					get show_fullscreen_button() {
						return get($0);
					},

					get fullscreen() {
						return fullscreen();
					},
					on_copy: handle_copy,
					get show_copy_button() {
						return get($1);
					},

					get show_search() {
						return show_search();
					},
					onsearch: (query) => handle_search(query),
					get onfullscreen() {
						return $$props.onfullscreen;
					},
					on_commit_filter: commit_filter,
					get current_search_query() {
						return get($2);
					}
				});
			}

			reset(div_1);
			append($$anchor, div_1);
		};

		if_block(node_1, ($$render) => {
			if (label() && label().length !== 0 && show_label() || (buttons() === null ? true : buttons().includes("fullscreen")) || (buttons() === null ? true : buttons().includes("copy")) || show_search() !== "none") $$render(consequent_1);
		});
	}

	var div_3 = sibling(node_1, 2);
	let classes_1;

	div_3.__keydown = handle_keydown;

	var node_4 = child(div_3);

	{
		let $0 = user_derived(() => $$props.i18n("dataframe.drop_to_upload"));

		Upload(node_4, {
			get upload() {
				return $$props.upload;
			},

			get stream_handler() {
				return $$props.stream_handler;
			},
			flex: false,
			center: false,
			boundedheight: false,
			disable_click: true,
			get root() {
				return $$props.root;
			},
			onload: on_file_upload,
			get aria_label() {
				return get($0);
			},

			get dragging() {
				return get(dragging);
			},

			set dragging($$value) {
				set(dragging, $$value, true);
			},

			children: ($$anchor, $$slotProps) => {
				var div_4 = root_4();
				let classes_2;
				var node_5 = child(div_4);

				{
					var consequent_2 = ($$anchor) => {
						var span = root_5();
						var text_1 = child(span, true);

						reset(span);
						template_effect(() => set_text(text_1, label()));
						append($$anchor, span);
					};

					if_block(node_5, ($$render) => {
						if (label() && label().length !== 0) $$render(consequent_2);
					});
				}

				var table_1 = sibling(node_5, 2);
				var thead = child(table_1);
				var tr = child(thead);
				var node_6 = child(tr);

				{
					var consequent_3 = ($$anchor) => {
						var th_1 = root_6();

						append($$anchor, th_1);
					};

					if_block(node_6, ($$render) => {
						if (show_row_numbers()) $$render(consequent_3);
					});
				}

				var node_7 = sibling(node_6);

				each(node_7, 17, () => get(header_groups), (headerGroup) => headerGroup.id, ($$anchor, headerGroup) => {
					var fragment_1 = comment();
					var node_8 = first_child(fragment_1);

					each(node_8, 17, () => get(headerGroup).headers, (header) => header.id, ($$anchor, header) => {
						const col_idx = user_derived(() => get(header).column.columnDef.meta?.colIndex ?? 0);

						{
							let $0 = user_derived(() => String(get(header).column.columnDef.header ?? ""));
							let $1 = user_derived(() => get(header_edit) === get(col_idx));
							let $2 = user_derived(() => get(selected_header) === get(col_idx));
							let $3 = user_derived(() => !!get(header).column.columnDef.meta?.isStatic);
							let $4 = user_derived(() => get_sort_info(get(col_idx)).direction);
							let $5 = user_derived(() => get_sort_info(get(col_idx)).priority);
							let $6 = user_derived(() => get(sorting).length > 1);
							let $7 = user_derived(() => get_filter_active(get(col_idx)));
							let $8 = user_derived(() => $$props.col_count[1] === "dynamic");
							let $9 = user_derived(() => get(col_idx) === 0 && !show_row_numbers());

							HeaderCell($$anchor, {
								get value() {
									return get($0);
								},

								get col_idx() {
									return get(col_idx);
								},

								get is_editing() {
									return get($1);
								},

								get is_selected() {
									return get($2);
								},

								get is_static() {
									return get($3);
								},

								get sort_direction() {
									return get($4);
								},

								get sort_priority() {
									return get($5);
								},

								get multi_sort() {
									return get($6);
								},

								get is_filtered() {
									return get($7);
								},

								get show_menu_button() {
									return get($8);
								},

								get is_first_column() {
									return get($9);
								},

								get latex_delimiters() {
									return $$props.latex_delimiters;
								},

								get line_breaks() {
									return line_breaks();
								},

								get editable() {
									return editable();
								},

								get max_chars() {
									return max_chars();
								},

								get i18n() {
									return $$props.i18n;
								},
								onclick: handle_header_click,
								on_menu_click: toggle_header_menu,
								on_end_edit: end_header_edit
							});
						}
					});

					append($$anchor, fragment_1);
				});

				reset(tr);
				bind_this(tr, ($$value) => header_row_el = $$value, () => header_row_el);
				reset(thead);

				var tbody = sibling(thead);
				var node_9 = child(tbody);

				{
					var consequent_6 = ($$anchor) => {
						const sizing_row = user_derived(() => get(rows).reduce(
							(widest, row) => {
								const cells = row.getVisibleCells();

								cells.forEach((cell, i) => {
									const val = String(cell.getValue() ?? "");

									if (!widest[i] || val.length > widest[i].length) {
										widest[i] = val;
									}
								});

								return widest;
							},
							[]
						));

						var tr_1 = root_9();
						var node_10 = child(tr_1);

						{
							var consequent_4 = ($$anchor) => {
								var td = root_10();
								var text_2 = child(td, true);

								reset(td);
								template_effect(() => set_text(text_2, get(rows).length));
								append($$anchor, td);
							};

							if_block(node_10, ($$render) => {
								if (show_row_numbers()) $$render(consequent_4);
							});
						}

						var node_11 = sibling(node_10);

						each(node_11, 17, () => get(sizing_row), index, ($$anchor, val, ci) => {
							const dtype = user_derived(() => get_dtype(ci));
							var td_1 = root_11();
							var div_5 = child(td_1);
							var node_12 = child(div_5);

							{
								var consequent_5 = ($$anchor) => {
									var fragment_3 = comment();
									var node_13 = first_child(fragment_3);

									html(node_13, () => get(val));
									append($$anchor, fragment_3);
								};

								var alternate = ($$anchor) => {
									var text_3 = text$1();

									template_effect(() => set_text(text_3, get(val)));
									append($$anchor, text_3);
								};

								if_block(node_12, ($$render) => {
									if (get(dtype) === "html" || get(dtype) === "markdown") $$render(consequent_5); else $$render(alternate, false);
								});
							}

							reset(div_5);
							reset(td_1);
							append($$anchor, td_1);
						});

						reset(tr_1);
						append($$anchor, tr_1);
					};

					if_block(node_9, ($$render) => {
						if (get(rows).length > 0) $$render(consequent_6);
					});
				}

				reset(tbody);
				reset(table_1);
				bind_this(table_1, ($$value) => header_table_el = $$value, () => header_table_el);

				var div_6 = sibling(table_1, 2);

				each(div_6, 21, () => get(virtual_items), (virtual_row) => virtual_row.key, ($$anchor, virtual_row) => {
					const row = user_derived(() => get(rows)[get(virtual_row).index]);
					const row_idx = user_derived(() => get(row)?.original._index ?? get(virtual_row).index);
					var fragment_5 = comment();
					var node_14 = first_child(fragment_5);

					{
						var consequent_8 = ($$anchor) => {
							var div_7 = root_15();
							let classes_3;
							var node_15 = child(div_7);

							{
								var consequent_7 = ($$anchor) => {
									var div_8 = root_16();
									var text_4 = child(div_8, true);

									reset(div_8);

									template_effect(() => {
										set_attribute(div_8, 'data-row', get(row_idx));
										set_style(div_8, `flex: 0 0 ${measurement.row_num_width ?? ''}px; width: ${measurement.row_num_width ?? ''}px;`);
										set_text(text_4, get(row_idx) + 1);
									});

									append($$anchor, div_8);
								};

								if_block(node_15, ($$render) => {
									if (show_row_numbers()) $$render(consequent_7);
								});
							}

							var node_16 = sibling(node_15, 2);

							each(node_16, 19, () => get(row).getVisibleCells(), (cell) => cell.id, ($$anchor, cell, ci) => {
								const col_idx = user_derived(() => get(cell).column.columnDef.meta?.colIndex ?? 0);
								const is_sel = user_derived(() => is_cell_in_selection([get(row_idx), get(col_idx)], get(selected_cells)));

								{
									let $0 = user_derived(() => get(cell).getValue());
									let $1 = user_derived(() => get_display_value(get(row_idx), get(col_idx)));
									let $2 = user_derived(() => get_dtype(get(col_idx)));
									let $3 = user_derived(() => measurement.get_col_style(get(ci)));
									let $4 = user_derived(() => get_styling(get(row_idx), get(col_idx)));
									let $5 = user_derived(() => is_cell_selected([get(row_idx), get(col_idx)], get(selected_cells)));
									let $6 = user_derived(() => !!(get(editing) && get(editing)[0] === get(row_idx) && get(editing)[1] === get(col_idx)));
									let $7 = user_derived(() => get(copy_flash) && get(is_sel));
									let $8 = user_derived(() => !!get(cell).column.columnDef.meta?.isStatic);
									let $9 = user_derived(() => editable() && get(selected_cells).length === 1 && get(selected_cells)[0][0] === get(row_idx) && get(selected_cells)[0][1] === get(col_idx));
									let $10 = user_derived(() => get(selected_cells).length === 1 && get(selected_cells)[0][0] === get(row_idx) && get(selected_cells)[0][1] === get(col_idx));
									let $11 = user_derived(() => get(ci) === 0 && !show_row_numbers());

									DataCell($$anchor, {
										get value() {
											return get($0);
										},

										get display_value() {
											return get($1);
										},

										get datatype() {
											return get($2);
										},

										get row_idx() {
											return get(row_idx);
										},

										get col_idx() {
											return get(col_idx);
										},

										get col_style() {
											return get($3);
										},

										get cell_style() {
											return get($4);
										},

										get selection_classes() {
											return get($5);
										},

										get is_editing() {
											return get($6);
										},

										get is_flash() {
											return get($7);
										},

										get is_static() {
											return get($8);
										},

										get show_menu_button() {
											return get($9);
										},

										get show_selection_buttons() {
											return get($10);
										},

										get is_first_column() {
											return get($11);
										},

										get latex_delimiters() {
											return $$props.latex_delimiters;
										},

										get line_breaks() {
											return line_breaks();
										},

										get editable() {
											return editable();
										},

										get max_chars() {
											return max_chars();
										},

										get i18n() {
											return $$props.i18n;
										},

										get components() {
											return components();
										},
										is_dragging,
										get wrap_text() {
											return wrap();
										},
										onmousedown: (e) => handle_cell_click(e, get(row_idx), get(col_idx)),
										ondblclick: (e) => handle_cell_dblclick(e, get(row_idx), get(col_idx)),
										oncontextmenu: (e) => {
											e.preventDefault();
											toggle_cell_menu(e, get(row_idx), get(col_idx));
										},
										onblur: handle_blur,
										on_menu_click: (e) => toggle_cell_menu(e, get(row_idx), get(col_idx)),
										on_select_column: (c) => {
											set(selected_cells, get(rows).map((_, r) => [r, c]), true);
											set(selected, get(selected_cells)[0], true);
										},

										on_select_row: (r) => {
											set(selected_cells, get(resolved_headers).map((_, c) => [r, c]), true);
											set(selected, get(selected_cells)[0], true);
										}
									});
								}
							});

							reset(div_7);
							action(div_7, ($$node, $$action_arg) => measure_row?.($$node), () => get(row));

							template_effect(
								($0) => {
									classes_3 = set_class(div_7, 1, 'virtual-row svelte-2balj6', null, classes_3, { 'row-odd': get(virtual_row).index % 2 !== 0 });
									set_attribute(div_7, 'data-index', get(virtual_row).index);
									set_style(div_7, `position: absolute; top: 0; left: 0; width: 100%; transform: translateY(${get(virtual_row).start ?? ''}px);${$0 ?? ''}`);
								},
								[
									() => get(selected_cells).some(([r]) => r === get(row_idx)) ? ' z-index: 3;' : ''
								]
							);

							append($$anchor, div_7);
						};

						if_block(node_14, ($$render) => {
							if (get(row)) $$render(consequent_8);
						});
					}

					append($$anchor, fragment_5);
				});

				reset(div_6);
				reset(div_4);
				bind_this(div_4, ($$value) => scroll_container = $$value, () => scroll_container);

				template_effect(() => {
					classes_2 = set_class(div_4, 1, 'virtual-table-viewport svelte-2balj6', null, classes_2, { 'disable-scroll': get(disable_scroll) });
					set_style(div_4, `max-height: ${max_height() ?? ''}px;`);
					set_style(div_6, `height: ${get(total_size) ?? ''}px; position: relative; flex-shrink: 0; width: ${measurement.total_header_width ? `${measurement.total_header_width}px` : '100%'};`);
				});

				event('scroll', div_4, handle_scroll);
				append($$anchor, div_4);
			},
			$$slots: { default: true }
		});
	}

	var node_17 = sibling(node_4, 2);

	{
		var consequent_9 = ($$anchor) => {
			var button = root_18();

			button.__click = scroll_to_top;
			append($$anchor, button);
		};

		if_block(node_17, ($$render) => {
			if (get(show_scroll_button)) $$render(consequent_9);
		});
	}

	reset(div_3);
	bind_this(div_3, ($$value) => parent = $$value, () => parent);
	reset(div);

	var node_18 = sibling(div, 2);

	{
		var consequent_10 = ($$anchor) => {
			{
				let $0 = user_derived(() => get(active_cell_menu)?.x ?? get(active_header_menu)?.x ?? 0);
				let $1 = user_derived(() => get(active_cell_menu)?.y ?? get(active_header_menu)?.y ?? 0);
				let $2 = user_derived(() => get(active_header_menu) ? -1 : get(active_cell_menu)?.row ?? 0);
				let $3 = user_derived(() => !get(active_header_menu) && values().length > 1 && editable());
				let $4 = user_derived(() => values().length > 0 && (values()[0]?.length ?? 0) > 1 && editable());

				let $5 = user_derived(() => get(active_header_menu)
					? (direction) => {
						handle_sort(get(active_header_menu).col, direction);
						set(active_header_menu, null);
					}
					: undefined);

				let $6 = user_derived(() => get(active_header_menu)
					? () => {
						clear_sort();
						set(active_header_menu, null);
					}
					: undefined);

				let $7 = user_derived(() => get(active_header_menu)
					? get_sort_info(get(active_header_menu).col).direction
					: null);

				let $8 = user_derived(() => get(active_header_menu)
					? get_sort_info(get(active_header_menu).col).priority
					: null);

				let $9 = user_derived(() => get(active_header_menu)
					? (dtype, filter, fvalue) => {
						handle_filter(get(active_header_menu).col, dtype, filter, fvalue);
						set(active_header_menu, null);
					}
					: undefined);

				let $10 = user_derived(() => get(active_header_menu)
					? () => {
						clear_filter();
						set(active_header_menu, null);
					}
					: undefined);

				let $11 = user_derived(() => get(active_header_menu)
					? get_filter_active(get(active_header_menu).col)
					: null);

				CellMenu($$anchor, {
					get x() {
						return get($0);
					},

					get y() {
						return get($1);
					},

					get row() {
						return get($2);
					},

					get col_count() {
						return $$props.col_count;
					},

					get row_count() {
						return $$props.row_count;
					},
					on_add_row_above: () => add_row_at(get(active_cell_menu)?.row ?? -1, "above"),
					on_add_row_below: () => add_row_at(get(active_cell_menu)?.row ?? -1, "below"),
					on_add_column_left: () => add_col_at(get(active_cell_menu)?.col ?? get(active_header_menu)?.col ?? -1, "left"),
					on_add_column_right: () => add_col_at(get(active_cell_menu)?.col ?? get(active_header_menu)?.col ?? -1, "right"),
					on_delete_row: () => delete_row_at(get(active_cell_menu)?.row ?? -1),
					on_delete_col: () => delete_col_at(get(active_cell_menu)?.col ?? get(active_header_menu)?.col ?? -1),
					get editable() {
						return editable();
					},

					get can_delete_rows() {
						return get($3);
					},

					get can_delete_cols() {
						return get($4);
					},

					get i18n() {
						return $$props.i18n;
					},

					get on_sort() {
						return get($5);
					},

					get on_clear_sort() {
						return get($6);
					},

					get sort_direction() {
						return get($7);
					},

					get sort_priority() {
						return get($8);
					},

					get on_filter() {
						return get($9);
					},

					get on_clear_filter() {
						return get($10);
					},

					get filter_active() {
						return get($11);
					}
				});
			}
		};

		if_block(node_18, ($$render) => {
			if (get(active_cell_menu) || get(active_header_menu)) $$render(consequent_10);
		});
	}

	var node_19 = sibling(node_18, 2);

	{
		var consequent_11 = ($$anchor) => {
			EmptyRowButton($$anchor, { on_click: () => add_row() });
		};

		if_block(node_19, ($$render) => {
			if (values().length === 0 && editable() && $$props.row_count[1] === "dynamic") $$render(consequent_11);
		});
	}

	template_effect(() => {
		classes = set_class(div, 1, 'table-container svelte-2balj6', null, classes, { fullscreen: fullscreen() });

		classes_1 = set_class(div_3, 1, 'table-wrap svelte-2balj6', null, classes_1, {
			dragging: is_dragging,
			'no-wrap': !wrap(),
			'menu-open': get(active_cell_menu) || get(active_header_menu)
		});
	});

	append($$anchor, fragment);
	pop();
}

delegate(['keydown', 'click']);

var root_1 = from_html(`<!> <!>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	let _props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(_props);
	let fullscreen = state(proxy(gradio.props.fullscreen ?? false));

	// align datatype array to current value headers using the original
	// config-time header→datatype mapping.
	// when columns are hidden or reordered, positional indices shift but
	// the datatype prop doesn't update, the map keeps them synced
	let aligned_datatype = user_derived(() => {
		const dt = gradio.props.datatype;

		if (!Array.isArray(dt)) return dt;

		const config_headers = gradio.props.headers;
		const current_headers = gradio.props.value?.headers;

		if (!config_headers || !current_headers) return dt;

		const map = new Map();

		for (let i = 0; i < Math.min(config_headers.length, dt.length); i++) {
			map.set(config_headers[i], dt[i]);
		}

		return current_headers.map((h, i) => map.get(h) ?? dt[i] ?? "str");
	});

	let old_value = state(proxy(gradio.props.value ? JSON.stringify(gradio.props.value) : null));

	function handle_change(detail) {
		gradio.props.value = detail;

		const serialized = JSON.stringify(detail);

		if (serialized !== get(old_value)) {
			set(old_value, serialized, true);
			gradio.dispatch("change", detail);
		}
	}

	function handle_input() {
		gradio.dispatch("input");
	}

	function handle_select(detail) {
		gradio.dispatch("select", detail);
	}

	function handle_edit(detail) {
		gradio.dispatch("edit", detail);
	}

	user_effect(() => {
		const v = gradio.props.value;

		if (v) {
			const serialized = JSON.stringify(v);

			if (serialized !== get(old_value)) {
				set(old_value, serialized, true);
				gradio.dispatch("change", v);
			}
		}
	});

	Block($$anchor, {
		get visible() {
			return gradio.shared.visible;
		},

		get elem_id() {
			return gradio.shared.elem_id;
		},

		get elem_classes() {
			return gradio.shared.elem_classes;
		},

		get scale() {
			return gradio.shared.scale;
		},

		get min_width() {
			return gradio.shared.min_width;
		},
		padding: false,
		container: false,
		overflow_behavior: 'visible',
		get fullscreen() {
			return get(fullscreen);
		},

		children: ($$anchor, $$slotProps) => {
			var fragment_1 = root_1();
			var node = first_child(fragment_1);

			Static(node, spread_props(
				{
					get autoscroll() {
						return gradio.shared.autoscroll;
					},

					get i18n() {
						return gradio.i18n;
					}
				},
				() => gradio.shared.loading_status
			));

			var node_1 = sibling(node, 2);

			{
				let $0 = user_derived(() => gradio.props.value?.headers ?? []);
				let $1 = user_derived(() => gradio.props.value?.data ?? []);
				let $2 = user_derived(() => gradio.props.value?.metadata?.display_value ?? null);
				let $3 = user_derived(() => gradio.props.value?.metadata?.styling ?? null);
				let $4 = user_derived(() => gradio.shared.interactive ?? true);
				let $5 = user_derived(() => gradio.props.column_widths ?? []);
				let $6 = user_derived(() => gradio.shared.client?.upload);
				let $7 = user_derived(() => gradio.shared.client?.stream);
				let $8 = user_derived(() => gradio.props.static_columns ?? []);

				Table(node_1, {
					get headers() {
						return get($0);
					},

					get values() {
						return get($1);
					},

					get display_value() {
						return get($2);
					},

					get styling() {
						return get($3);
					},

					get col_count() {
						return gradio.props.col_count;
					},

					get row_count() {
						return gradio.props.row_count;
					},

					get label() {
						return gradio.shared.label;
					},

					get show_label() {
						return gradio.shared.show_label;
					},

					get wrap() {
						return gradio.props.wrap;
					},

					get datatype() {
						return get(aligned_datatype);
					},

					get latex_delimiters() {
						return gradio.props.latex_delimiters;
					},

					get max_height() {
						return gradio.props.max_height;
					},

					get editable() {
						return get($4);
					},

					get line_breaks() {
						return gradio.props.line_breaks;
					},

					get column_widths() {
						return get($5);
					},

					get root() {
						return gradio.shared.root;
					},

					get i18n() {
						return gradio.i18n;
					},

					get upload() {
						return get($6);
					},

					get stream_handler() {
						return get($7);
					},

					get buttons() {
						return gradio.props.buttons;
					},

					get max_chars() {
						return gradio.props.max_chars;
					},

					get show_row_numbers() {
						return gradio.props.show_row_numbers;
					},

					get show_search() {
						return gradio.props.show_search;
					},

					get pinned_columns() {
						return gradio.props.pinned_columns;
					},

					get static_columns() {
						return get($8);
					},

					get fullscreen() {
						return get(fullscreen);
					},

					onfullscreen: () => {
						set(fullscreen, !get(fullscreen));
					},
					onchange: handle_change,
					oninput: handle_input,
					onselect: handle_select,
					onedit: handle_edit
				});
			}

			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	pop();
}

export { Table as BaseDataFrame, Index as default };
//# sourceMappingURL=Index-CxOOE7R5.js.map
