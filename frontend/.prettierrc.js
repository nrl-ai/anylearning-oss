module.exports = {
    // this line is required for pnpm
    plugins: ["@trivago/prettier-plugin-sort-imports", "prettier-plugin-tailwindcss"],

    semi: false,
    tabWidth: 4,
    printWidth: 120,
    singleQuote: false,
    trailingComma: "es5",
    bracketSameLine: false,
    endOfLine: "lf",
    overrides: [
        {
            files: "*.yml",
            options: {
                tabWidth: 2,
            },
        },
        {
            files: "*.yaml",
            options: {
                tabWidth: 2,
            },
        },
    ],
    importOrder: [
        "<THIRD_PARTY_MODULES>",
        "^@/(app|assets|components|configs|constants|domains|hoc|hooks|lib|packages|requests|sections|store|types)(/.+)?$",
        "\\.(c|le|sc)ss$",
        "^[./]",
    ],
    importOrderSeparation: true,
    importOrderSortSpecifiers: true,
}
