def center_window(win, width, height):
    """Centers a tkinter window on the screen."""
    # Update idle tasks to ensure accurate window size calculations
    win.update_idletasks()
    
    # Get the screen's width and height
    screen_width = win.winfo_screenwidth()
    screen_height = win.winfo_screenheight()

    # Calculate the x and y coordinates for the top-left corner
    # The calculations account for the window's title bar and borders for better accuracy
    # Simple calculation for center position
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    
    # Set the geometry (widthxheight+x+y)
    win.geometry(f'{width}x{height}+{x}+{y}')