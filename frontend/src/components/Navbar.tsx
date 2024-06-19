import React from "react";
import { Button } from "./ui/button";
import { ModeToggle } from "./ModeToggle";

const Navbar = () => {
  return (
    <nav className="max-w-4xl p-3 mx-auto flex items-center justify-between border-b border-gray-500">
      <h1 className="text-2xl">ArtiStyle</h1>
      <div className="flex items-center gap-3">
        <Button>Login</Button>
        <ModeToggle />
      </div>
    </nav>
  );
};

export default Navbar;
