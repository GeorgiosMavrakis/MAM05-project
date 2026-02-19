import { FC } from "react";
import { AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface ErrorMessageProps {
  content: string;
  className?: string;
}

export const ErrorMessage: FC<ErrorMessageProps> = ({ content, className }) => {
  return (
    <div
      className={cn(
        "aui-error-message-root fade-in slide-in-from-bottom-1 relative mx-auto w-full max-w-[44rem] animate-in py-3 duration-150",
        className,
      )}
    >
      <div className="aui-error-message-content mx-2 rounded-lg border-destructive border-l-4 bg-destructive/10 p-4 text-destructive dark:bg-destructive/5 dark:text-red-200">
        <div className="flex items-start gap-3">
          <AlertCircle className="mt-0.5 h-5 w-5 flex-shrink-0" />
          <div className="flex-1">
            <p className="whitespace-pre-wrap break-words text-sm leading-relaxed">
              {content}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
