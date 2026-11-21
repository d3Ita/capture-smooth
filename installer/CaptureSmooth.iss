; Windows installer for Capture Smooth (Inno Setup 6).
;   ISCC /DEdition=NVIDIA installer\CaptureSmooth.iss     -> dist\installer\CaptureSmooth-<ver>-NVIDIA-Setup.exe (+ .bin parts)
;   ISCC /DEdition=DirectML installer\CaptureSmooth.iss
; Installs for the current user (no administrator prompt). The NVIDIA edition is split in parts under
; GitHub's 2 GB per-file limit: keep Setup.exe and its .bin files in the same folder.

#ifndef Edition
  #define Edition "NVIDIA"
#endif
#define AppVer "1.4.0"
#define ExeName "CaptureSmooth-" + Edition + ".exe"

[Setup]
#if Edition == "NVIDIA"
AppId={{6E3C2B1A-4F0D-4C55-9C21-5A7E0B9D1A01}
DefaultDirName={localappdata}\Programs\Capture Smooth
DiskSpanning=yes
DiskSliceSize=1900000000
#else
AppId={{6E3C2B1A-4F0D-4C55-9C21-5A7E0B9D1A02}
DefaultDirName={localappdata}\Programs\Capture Smooth DirectML
#endif
AppName=Capture Smooth
AppVersion={#AppVer}
AppVerName=Capture Smooth {#AppVer} ({#Edition})
AppPublisher=Capture Smooth (open source)
AppPublisherURL=https://d3ita.github.io/capture-smooth/
AppSupportURL=https://github.com/d3Ita/capture-smooth/issues
AppUpdatesURL=https://github.com/d3Ita/capture-smooth/releases
LicenseFile=..\LICENSE
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
OutputDir=..\dist\installer
OutputBaseFilename=CaptureSmooth-{#AppVer}-{#Edition}-Setup
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\{#ExeName}
UninstallDisplayName=Capture Smooth ({#Edition})
Compression=lzma2/fast
SolidCompression=no
LZMAUseSeparateProcess=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes

[Languages]
Name: "french"; MessagesFile: "compiler:Languages\French.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\CaptureSmooth-{#Edition}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Capture Smooth"; Filename: "{app}\{#ExeName}"
Name: "{autoprograms}\Capture Smooth - réafficher la barre des tâches"; Filename: "{app}\{#ExeName}"; Parameters: "--show-taskbar"
Name: "{autodesktop}\Capture Smooth"; Filename: "{app}\{#ExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#ExeName}"; Description: "{cm:LaunchProgram,Capture Smooth}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\_internal"
