"use strict";
/*
 * ATTENTION: An "eval-source-map" devtool has been used.
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file with attached SourceMaps in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
exports.id = "vendor-chunks/hast-util-whitespace";
exports.ids = ["vendor-chunks/hast-util-whitespace"];
exports.modules = {

/***/ "(ssr)/./node_modules/hast-util-whitespace/lib/index.js":
/*!********************************************************!*\
  !*** ./node_modules/hast-util-whitespace/lib/index.js ***!
  \********************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

eval("__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   whitespace: () => (/* binding */ whitespace)\n/* harmony export */ });\n/**\n * @typedef {import('hast').Nodes} Nodes\n */\n\n// HTML whitespace expression.\n// See <https://infra.spec.whatwg.org/#ascii-whitespace>.\nconst re = /[ \\t\\n\\f\\r]/g\n\n/**\n * Check if the given value is *inter-element whitespace*.\n *\n * @param {Nodes | string} thing\n *   Thing to check (`Node` or `string`).\n * @returns {boolean}\n *   Whether the `value` is inter-element whitespace (`boolean`): consisting of\n *   zero or more of space, tab (`\\t`), line feed (`\\n`), carriage return\n *   (`\\r`), or form feed (`\\f`); if a node is passed it must be a `Text` node,\n *   whose `value` field is checked.\n */\nfunction whitespace(thing) {\n  return typeof thing === 'object'\n    ? thing.type === 'text'\n      ? empty(thing.value)\n      : false\n    : empty(thing)\n}\n\n/**\n * @param {string} value\n * @returns {boolean}\n */\nfunction empty(value) {\n  return value.replace(re, '') === ''\n}\n//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiKHNzcikvLi9ub2RlX21vZHVsZXMvaGFzdC11dGlsLXdoaXRlc3BhY2UvbGliL2luZGV4LmpzIiwibWFwcGluZ3MiOiI7Ozs7QUFBQTtBQUNBLGFBQWEsc0JBQXNCO0FBQ25DOztBQUVBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQSxXQUFXLGdCQUFnQjtBQUMzQjtBQUNBLGFBQWE7QUFDYjtBQUNBO0FBQ0Esa0NBQWtDO0FBQ2xDO0FBQ0E7QUFDTztBQUNQO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBLFdBQVcsUUFBUTtBQUNuQixhQUFhO0FBQ2I7QUFDQTtBQUNBO0FBQ0EiLCJzb3VyY2VzIjpbIi9Vc2Vycy9hamVvbi9zb3RvcGlhL2V4YW1wbGVzL2V4cGVyaW1lbnRhbC9ub2Rlcy9mcm9udGVuZC9ub2RlX21vZHVsZXMvaGFzdC11dGlsLXdoaXRlc3BhY2UvbGliL2luZGV4LmpzIl0sInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQHR5cGVkZWYge2ltcG9ydCgnaGFzdCcpLk5vZGVzfSBOb2Rlc1xuICovXG5cbi8vIEhUTUwgd2hpdGVzcGFjZSBleHByZXNzaW9uLlxuLy8gU2VlIDxodHRwczovL2luZnJhLnNwZWMud2hhdHdnLm9yZy8jYXNjaWktd2hpdGVzcGFjZT4uXG5jb25zdCByZSA9IC9bIFxcdFxcblxcZlxccl0vZ1xuXG4vKipcbiAqIENoZWNrIGlmIHRoZSBnaXZlbiB2YWx1ZSBpcyAqaW50ZXItZWxlbWVudCB3aGl0ZXNwYWNlKi5cbiAqXG4gKiBAcGFyYW0ge05vZGVzIHwgc3RyaW5nfSB0aGluZ1xuICogICBUaGluZyB0byBjaGVjayAoYE5vZGVgIG9yIGBzdHJpbmdgKS5cbiAqIEByZXR1cm5zIHtib29sZWFufVxuICogICBXaGV0aGVyIHRoZSBgdmFsdWVgIGlzIGludGVyLWVsZW1lbnQgd2hpdGVzcGFjZSAoYGJvb2xlYW5gKTogY29uc2lzdGluZyBvZlxuICogICB6ZXJvIG9yIG1vcmUgb2Ygc3BhY2UsIHRhYiAoYFxcdGApLCBsaW5lIGZlZWQgKGBcXG5gKSwgY2FycmlhZ2UgcmV0dXJuXG4gKiAgIChgXFxyYCksIG9yIGZvcm0gZmVlZCAoYFxcZmApOyBpZiBhIG5vZGUgaXMgcGFzc2VkIGl0IG11c3QgYmUgYSBgVGV4dGAgbm9kZSxcbiAqICAgd2hvc2UgYHZhbHVlYCBmaWVsZCBpcyBjaGVja2VkLlxuICovXG5leHBvcnQgZnVuY3Rpb24gd2hpdGVzcGFjZSh0aGluZykge1xuICByZXR1cm4gdHlwZW9mIHRoaW5nID09PSAnb2JqZWN0J1xuICAgID8gdGhpbmcudHlwZSA9PT0gJ3RleHQnXG4gICAgICA/IGVtcHR5KHRoaW5nLnZhbHVlKVxuICAgICAgOiBmYWxzZVxuICAgIDogZW1wdHkodGhpbmcpXG59XG5cbi8qKlxuICogQHBhcmFtIHtzdHJpbmd9IHZhbHVlXG4gKiBAcmV0dXJucyB7Ym9vbGVhbn1cbiAqL1xuZnVuY3Rpb24gZW1wdHkodmFsdWUpIHtcbiAgcmV0dXJuIHZhbHVlLnJlcGxhY2UocmUsICcnKSA9PT0gJydcbn1cbiJdLCJuYW1lcyI6W10sImlnbm9yZUxpc3QiOlswXSwic291cmNlUm9vdCI6IiJ9\n//# sourceURL=webpack-internal:///(ssr)/./node_modules/hast-util-whitespace/lib/index.js\n");

/***/ })

};
;