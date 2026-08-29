import { Fragment } from "react"

import { cn } from "@/lib/utils"

/**
 * A small markdown renderer for the documents the app ships.
 *
 * Not react-markdown: legal documents in this application use only a small
 * syntax is headings, bold, links, lists, block quotes, code spans and rules.
 * A dependency (plus remark, plus its plugin graph) to render six constructs is
 * a poor trade in an app that already ships 3 GB and has to build offline.
 *
 * Elements are constructed rather than injected as HTML, so a document can
 * never introduce markup -- there is no dangerouslySetInnerHTML here and there
 * should not be one.
 */

/** Bold, code spans and links, in that order of precedence. */
function inline(text: string, keyPrefix: string) {
    const pattern = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g
    const parts = text.split(pattern).filter(Boolean)

    return parts.map((part, index) => {
        const key = `${keyPrefix}-${index}`
        if (part.startsWith("**") && part.endsWith("**")) {
            return (
                <strong key={key} className="text-foreground font-medium">
                    {part.slice(2, -2)}
                </strong>
            )
        }
        if (part.startsWith("`") && part.endsWith("`")) {
            return (
                <code key={key} className="bg-muted rounded px-1 py-0.5 font-mono text-[0.9em]">
                    {part.slice(1, -1)}
                </code>
            )
        }
        const link = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(part)
        if (link) {
            const href = link[2]
            // Only http(s): a document should not be able to produce a
            // javascript: or file: link, however it got here.
            const safe = /^https?:\/\//i.test(href)
            return safe ? (
                <a
                    key={key}
                    href={href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-foreground underline underline-offset-4"
                >
                    {link[1]}
                </a>
            ) : (
                <Fragment key={key}>{link[1]}</Fragment>
            )
        }
        return <Fragment key={key}>{part}</Fragment>
    })
}

export function Markdown({ text, className }: { text: string; className?: string }) {
    const blocks: React.ReactNode[] = []
    const lines = text.split("\n")
    let list: string[] = []
    let quote: string[] = []
    // Markdown wraps: consecutive non-empty lines are one paragraph, and
    // rendering each source line separately turned every sentence into a
    // ladder of short lines.
    let paragraph: string[] = []
    let table: string[] = []

    const flushList = () => {
        if (list.length === 0) return
        blocks.push(
            <ul key={`ul-${blocks.length}`} className="list-disc space-y-1 pl-5">
                {list.map((item, index) => (
                    <li key={index}>{inline(item, `li-${blocks.length}-${index}`)}</li>
                ))}
            </ul>
        )
        list = []
    }

    const flushQuote = () => {
        if (quote.length === 0) return
        blocks.push(
            <blockquote
                key={`quote-${blocks.length}`}
                className="border-warn-border bg-warn-surface text-warn rounded-md border-l-2 px-3 py-2"
            >
                {inline(quote.join(" "), `q-${blocks.length}`)}
            </blockquote>
        )
        quote = []
    }

    const flushTable = () => {
        if (table.length === 0) return
        const rows = table
            .map((row) =>
                row
                    .replace(/^\||\|$/g, "")
                    .split("|")
                    .map((cell) => cell.trim())
            )
            // The |---|---| separator is layout, not content.
            .filter((cells) => !cells.every((cell) => /^:?-{2,}:?$/.test(cell)))
        const [header, ...body] = rows
        blocks.push(
            <div key={`table-${blocks.length}`} className="overflow-x-auto">
                <table className="w-full border-collapse text-left">
                    <thead>
                        <tr className="border-b">
                            {header.map((cell, index) => (
                                <th key={index} className="text-foreground py-1 pr-3 font-medium">
                                    {inline(cell, `th-${blocks.length}-${index}`)}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {body.map((cells, rowIndex) => (
                            <tr key={rowIndex} className="border-b last:border-b-0">
                                {cells.map((cell, index) => (
                                    <td key={index} className="py-1 pr-3 align-top">
                                        {inline(cell, `td-${blocks.length}-${rowIndex}-${index}`)}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        )
        table = []
    }

    const flushParagraph = () => {
        if (paragraph.length === 0) return
        blocks.push(<p key={`p-${blocks.length}`}>{inline(paragraph.join(" "), `p-${blocks.length}`)}</p>)
        paragraph = []
    }

    lines.forEach((raw, index) => {
        const line = raw.trimEnd()

        if (line.trim().startsWith("|") && line.trim().endsWith("|")) {
            flushParagraph()
            flushList()
            table.push(line.trim())
            return
        }
        flushTable()

        if (line.startsWith(">")) {
            flushList()
            flushParagraph()
            const content = line.replace(/^>\s?/, "")
            if (content) quote.push(content)
            return
        }
        flushQuote()

        if (/^[-*]\s+/.test(line)) {
            flushParagraph()
            list.push(line.replace(/^[-*]\s+/, ""))
            return
        }
        flushList()

        if (!line.trim()) {
            flushParagraph()
            return
        }

        if (line.startsWith("### ")) {
            flushParagraph()
            blocks.push(
                <h3 key={index} className="text-foreground pt-2 text-sm font-medium">
                    {inline(line.slice(4), `h3-${index}`)}
                </h3>
            )
        } else if (line.startsWith("## ")) {
            flushParagraph()
            blocks.push(
                <h2 key={index} className="text-foreground pt-3 text-sm font-semibold">
                    {inline(line.slice(3), `h2-${index}`)}
                </h2>
            )
        } else if (line.startsWith("# ")) {
            flushParagraph()
            blocks.push(
                <h1 key={index} className="text-foreground text-base font-semibold">
                    {inline(line.slice(2), `h1-${index}`)}
                </h1>
            )
        } else if (/^-{3,}$/.test(line.trim())) {
            flushParagraph()
            blocks.push(<hr key={index} className="border-border" />)
        } else {
            paragraph.push(line)
        }
    })

    flushList()
    flushQuote()
    flushParagraph()
    flushTable()

    return <div className={cn("space-y-2 text-xs leading-relaxed", className)}>{blocks}</div>
}
