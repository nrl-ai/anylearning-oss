import styles from "./style.module.css";

export function OptionTable({ options }: { options: [string, string, any] }) {
  return (
    <div
      className={
        "-mx-6 mt-6 mb-4 overflow-x-auto overscroll-x-contain px-6 pb-4 " +
        styles.container
      }
    >
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-line py-4 text-left">
            <th className="py-2 font-semibold">Option</th>
            <th className="py-2 pl-6 font-semibold">Type</th>
            <th className="py-2 px-6 font-semibold">Description</th>
          </tr>
        </thead>
        <tbody className="align-baseline text-foreground">
          {options.map(([option, type, description]) => (
            <tr key={option} className="border-b border-line">
              <td className="whitespace-pre py-2 font-mono tabular text-xs font-medium leading-6 text-mark">
                {option}
              </td>
              <td className="whitespace-pre py-2 pl-6 font-mono tabular text-xs leading-6 text-muted-foreground">
                {type}
              </td>
              <td className="py-2 pl-6">{description}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
