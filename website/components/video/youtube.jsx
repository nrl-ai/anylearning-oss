import dynamic from "next/dynamic";
const ReactPlayer = dynamic(() => import("react-player"), { ssr: false });

export default function YouTube({ ...props }) {
  return (
    <div className="mr-auto ml-auto aspect-[16/9] h-auto w-[100%] md:max-w-[100%] mt-2 mb-2">
      <ReactPlayer {...props} width={"100%"} height={"100%"} />
    </div>
  );
}
