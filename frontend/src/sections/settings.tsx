"use client"

import { Gauge, Info, Monitor, Moon, Sun, Tag } from "lucide-react"
import { useTheme } from "next-themes"

import { LegalNotices } from "@/components/legal-notices"
import { useSettingStore } from "@/components/react-image-label/base/store"
import { ModelLicences, SoftwareLicense } from "@/components/terms-of-use"
import { Button } from "@/components/ui/button"
import { Panel, PanelBody, PanelHeader, Stat } from "@/components/ui/panel"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { APP_VERSION, DATA_ROOT, PRODUCT_NAME } from "@/lib/app-info"
import { PerformanceMode, useAppSettings } from "@/lib/use-app-settings"
import useMounted from "@/lib/use-mounted"
import { usePreferences } from "@/lib/use-preferences"
import { useAutoSaveSettingStore } from "@/sections/labelling/stores"

/** One labelled control row. Keeps every setting on the same rhythm. */
function Setting({ label, description, control }: { label: string; description?: string; control: React.ReactNode }) {
    return (
        <div className="flex items-center justify-between gap-6 py-3 first:pt-0 last:pb-0">
            <div className="min-w-0">
                <p className="text-sm font-medium">{label}</p>
                {description && <p className="text-muted-foreground mt-0.5 text-xs">{description}</p>}
            </div>
            <div className="shrink-0">{control}</div>
        </div>
    )
}

const THEMES = [
    { value: "light", label: "Light", icon: Sun },
    { value: "dark", label: "Dark", icon: Moon },
    { value: "system", label: "System", icon: Monitor },
]

export default function Settings() {
    const mounted = useMounted()
    const { theme, setTheme } = useTheme()
    const { isEnabled: autoSave, setEnabled: setAutoSave } = useAutoSaveSettingStore()
    const { isShowLabels, setShowLabels } = useSettingStore()
    const { settings: perf, setPerformanceMode, isSaving: isSavingPerf } = useAppSettings()
    const { gridPageSize, setGridPageSize, modelsPageSize, setModelsPageSize } = usePreferences()

    // Every control below reflects stored state, which only exists in the
    // browser. Rendering it before mount would disagree with the prerendered
    // HTML -- see use-mounted.
    if (!mounted) return null

    return (
        <div className="mx-auto grid w-full max-w-3xl gap-4">
            <Panel>
                <PanelHeader icon={Sun} title="Appearance" />
                <PanelBody className="divide-y">
                    <Setting
                        label="Theme"
                        description="Dark suits long labelling sessions; the chrome stays neutral either way."
                        control={
                            <div className="bg-muted flex gap-0.5 rounded-md p-0.5">
                                {THEMES.map(({ value, label, icon: Icon }) => (
                                    <Button
                                        key={value}
                                        variant="ghost"
                                        size="sm"
                                        aria-pressed={theme === value}
                                        onClick={() => setTheme(value)}
                                        className={
                                            theme === value
                                                ? "bg-surface text-foreground shadow-xs"
                                                : "text-muted-foreground"
                                        }
                                    >
                                        <Icon />
                                        {label}
                                    </Button>
                                ))}
                            </div>
                        }
                    />
                </PanelBody>
            </Panel>

            <Panel>
                <PanelHeader icon={Tag} title="Labelling" />
                <PanelBody className="divide-y">
                    <Setting
                        label="Auto-save annotations"
                        description="Saves the current image before moving to the next one."
                        control={<Switch checked={autoSave} onCheckedChange={setAutoSave} aria-label="Auto-save" />}
                    />
                    <Setting
                        label="Show class names on the canvas"
                        description="Draws each shape's class next to it while labelling."
                        control={
                            <Switch
                                checked={isShowLabels}
                                onCheckedChange={setShowLabels}
                                aria-label="Show class names"
                            />
                        }
                    />
                </PanelBody>
            </Panel>

            <Panel>
                <PanelHeader icon={Gauge} title="Performance" />
                <PanelBody className="divide-y">
                    <Setting
                        label="Training performance"
                        description={
                            perf
                                ? `How hard to work this machine while training. ${perf.resolved.physical_cores} cores ` +
                                  `(${perf.resolved.cpu_count} threads) available; currently ` +
                                  `${perf.resolved.num_workers_gpu} data-loading workers with a GPU, ` +
                                  `${perf.resolved.num_workers_cpu} without.`
                                : "How hard to work this machine while training."
                        }
                        control={
                            <Select
                                value={perf?.performance_mode ?? "maximum"}
                                onValueChange={(v) => setPerformanceMode(v as PerformanceMode)}
                                disabled={!perf || isSavingPerf}
                            >
                                <SelectTrigger size="sm" className="w-[152px]">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="maximum">Maximum</SelectItem>
                                    <SelectItem value="balanced">Balanced</SelectItem>
                                    <SelectItem value="power_saving">Power saving</SelectItem>
                                </SelectContent>
                            </Select>
                        }
                    />
                </PanelBody>
            </Panel>

            <Panel>
                <PanelHeader icon={Monitor} title="Lists" />
                <PanelBody className="divide-y">
                    <Setting
                        label="Images per page"
                        description="How many thumbnails the dataset grid loads at a time."
                        control={
                            <Select value={String(gridPageSize)} onValueChange={(v) => setGridPageSize(Number(v))}>
                                <SelectTrigger size="sm" className="w-[84px]">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    {[10, 20, 50, 100].map((n) => (
                                        <SelectItem key={n} value={String(n)}>
                                            {n}
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        }
                    />
                    <Setting
                        label="Models per page"
                        control={
                            <Select value={String(modelsPageSize)} onValueChange={(v) => setModelsPageSize(Number(v))}>
                                <SelectTrigger size="sm" className="w-[84px]">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    {[5, 10, 20, 50].map((n) => (
                                        <SelectItem key={n} value={String(n)}>
                                            {n}
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        }
                    />
                </PanelBody>
            </Panel>

            <Panel>
                <PanelHeader icon={Info} title="About" />
                <PanelBody>
                    <div className="grid gap-4 sm:grid-cols-3">
                        <Stat label="Product" value={PRODUCT_NAME} mono={false} />
                        <Stat label="Version" value={APP_VERSION} />
                        <Stat label="Data folder" value={DATA_ROOT} hint="Projects, models and databases live here." />
                    </div>
                    <div className="mt-4 flex flex-wrap gap-2">
                        <SoftwareLicense />
                        <LegalNotices />
                        <ModelLicences />
                    </div>
                </PanelBody>
            </Panel>
        </div>
    )
}
