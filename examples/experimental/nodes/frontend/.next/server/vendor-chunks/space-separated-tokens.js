"use strict";
/*
 * ATTENTION: An "eval-source-map" devtool has been used.
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file with attached SourceMaps in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
exports.id = "vendor-chunks/space-separated-tokens";
exports.ids = ["vendor-chunks/space-separated-tokens"];
exports.modules = {

/***/ "(ssr)/./node_modules/space-separated-tokens/index.js":
/*!******************************************************!*\
  !*** ./node_modules/space-separated-tokens/index.js ***!
  \******************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

eval("__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   parse: () => (/* binding */ parse),\n/* harmony export */   stringify: () => (/* binding */ stringify)\n/* harmony export */ });\n/**\n * Parse space-separated tokens to an array of strings.\n *\n * @param {string} value\n *   Space-separated tokens.\n * @returns {Array<string>}\n *   List of tokens.\n */\nfunction parse(value) {\n  const input = String(value || '').trim()\n  return input ? input.split(/[ \\t\\n\\r\\f]+/g) : []\n}\n\n/**\n * Serialize an array of strings as space separated-tokens.\n *\n * @param {Array<string|number>} values\n *   List of tokens.\n * @returns {string}\n *   Space-separated tokens.\n */\nfunction stringify(values) {\n  return values.join(' ').trim()\n}\n//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiKHNzcikvLi9ub2RlX21vZHVsZXMvc3BhY2Utc2VwYXJhdGVkLXRva2Vucy9pbmRleC5qcyIsIm1hcHBpbmdzIjoiOzs7OztBQUFBO0FBQ0E7QUFDQTtBQUNBLFdBQVcsUUFBUTtBQUNuQjtBQUNBLGFBQWE7QUFDYjtBQUNBO0FBQ087QUFDUDtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0EsV0FBVyxzQkFBc0I7QUFDakM7QUFDQSxhQUFhO0FBQ2I7QUFDQTtBQUNPO0FBQ1A7QUFDQSIsInNvdXJjZXMiOlsiL1VzZXJzL2FqZW9uL3NvdG9waWEvZXhhbXBsZXMvZXhwZXJpbWVudGFsL25vZGVzL2Zyb250ZW5kL25vZGVfbW9kdWxlcy9zcGFjZS1zZXBhcmF0ZWQtdG9rZW5zL2luZGV4LmpzIl0sInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogUGFyc2Ugc3BhY2Utc2VwYXJhdGVkIHRva2VucyB0byBhbiBhcnJheSBvZiBzdHJpbmdzLlxuICpcbiAqIEBwYXJhbSB7c3RyaW5nfSB2YWx1ZVxuICogICBTcGFjZS1zZXBhcmF0ZWQgdG9rZW5zLlxuICogQHJldHVybnMge0FycmF5PHN0cmluZz59XG4gKiAgIExpc3Qgb2YgdG9rZW5zLlxuICovXG5leHBvcnQgZnVuY3Rpb24gcGFyc2UodmFsdWUpIHtcbiAgY29uc3QgaW5wdXQgPSBTdHJpbmcodmFsdWUgfHwgJycpLnRyaW0oKVxuICByZXR1cm4gaW5wdXQgPyBpbnB1dC5zcGxpdCgvWyBcXHRcXG5cXHJcXGZdKy9nKSA6IFtdXG59XG5cbi8qKlxuICogU2VyaWFsaXplIGFuIGFycmF5IG9mIHN0cmluZ3MgYXMgc3BhY2Ugc2VwYXJhdGVkLXRva2Vucy5cbiAqXG4gKiBAcGFyYW0ge0FycmF5PHN0cmluZ3xudW1iZXI+fSB2YWx1ZXNcbiAqICAgTGlzdCBvZiB0b2tlbnMuXG4gKiBAcmV0dXJucyB7c3RyaW5nfVxuICogICBTcGFjZS1zZXBhcmF0ZWQgdG9rZW5zLlxuICovXG5leHBvcnQgZnVuY3Rpb24gc3RyaW5naWZ5KHZhbHVlcykge1xuICByZXR1cm4gdmFsdWVzLmpvaW4oJyAnKS50cmltKClcbn1cbiJdLCJuYW1lcyI6W10sImlnbm9yZUxpc3QiOlswXSwic291cmNlUm9vdCI6IiJ9\n//# sourceURL=webpack-internal:///(ssr)/./node_modules/space-separated-tokens/index.js\n");

/***/ })

};
;