"use client";

import { useState, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { AlertTriangle } from "lucide-react";

export function DisclaimerDialog() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    // Check if user has already accepted disclaimer
    const hasAccepted = localStorage.getItem("disclaimer-accepted");
    if (!hasAccepted) {
      setOpen(true);
    }
  }, []);

  const handleAccept = () => {
    localStorage.setItem("disclaimer-accepted", "true");
    setOpen(false);
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="sm:max-w-125" showCloseButton={false}>
        <DialogHeader>
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-6 w-6 text-yellow-500" />
            <DialogTitle className="text-xl">OpenFDA Disclaimer</DialogTitle>
          </div>
        </DialogHeader>
        <div className="text-left pt-4 space-y-3">
          <p className="text-base font-semibold text-foreground">
            Do not rely on Medication Assistant to make decisions regarding medical care.
          </p>
          <p className="text-sm text-muted-foreground">
            While we make every effort to ensure that data is accurate, you should assume all results are unvalidated.
          </p>
          <p className="text-sm text-muted-foreground">
            This information is provided for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
          </p>
        </div>
        <DialogFooter>
          <Button onClick={handleAccept} className="w-full sm:w-auto">
            I Understand
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}


