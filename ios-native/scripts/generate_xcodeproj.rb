#!/usr/bin/env ruby
# frozen_string_literal: true

require "xcodeproj"
require "fileutils"

ROOT = File.expand_path("..", __dir__)
PROJECT_PATH = File.join(ROOT, "SwingSenseiNative.xcodeproj")

FileUtils.rm_rf(PROJECT_PATH)

project = Xcodeproj::Project.new(PROJECT_PATH)
project.instance_variable_set(:@object_version, "60")
project.root_object.attributes["LastSwiftUpdateCheck"] = "2630"
project.root_object.attributes["LastUpgradeCheck"] = "2630"

app_target = project.new_target(:application, "SwingSenseiNative", :ios, "17.0")
project.root_object.attributes["TargetAttributes"] ||= {}
project.root_object.attributes["TargetAttributes"][app_target.uuid] ||= {}
project.root_object.attributes["TargetAttributes"][app_target.uuid]["DevelopmentTeam"] = "9974PM5L6M"

main_group = project.main_group
app_group = main_group.new_group("SwingSenseiNative", "SwingSenseiNative")
core_group = main_group.new_group("SwingSenseiCore", "Sources/SwingSenseiCore")

app_sources = Dir[File.join(ROOT, "SwingSenseiNative", "*.swift")].sort
core_sources = Dir[File.join(ROOT, "Sources", "SwingSenseiCore", "*.swift")].sort

(app_sources + core_sources).each do |path|
  relative = path.sub("#{ROOT}/", "")
  group = relative.start_with?("Sources/") ? core_group : app_group
  file = group.new_file(File.basename(relative))
  app_target.add_file_references([file])
end

info_plist = app_group.new_file("Info.plist")

app_target.build_configurations.each do |config|
  settings = config.build_settings
  settings["ASSETCATALOG_COMPILER_GLOBAL_ACCENT_COLOR_NAME"] = "AccentColor"
  settings["CODE_SIGN_STYLE"] = "Automatic"
  settings["CURRENT_PROJECT_VERSION"] = "1"
  settings["DEVELOPMENT_TEAM"] = "9974PM5L6M"
  settings["ENABLE_PREVIEWS"] = "YES"
  settings["GENERATE_INFOPLIST_FILE"] = "NO"
  settings["INFOPLIST_FILE"] = "SwingSenseiNative/Info.plist"
  settings["IPHONEOS_DEPLOYMENT_TARGET"] = "17.0"
  settings["LD_RUNPATH_SEARCH_PATHS"] = "$(inherited) @executable_path/Frameworks"
  settings["MARKETING_VERSION"] = "1.0"
  settings["PRODUCT_BUNDLE_IDENTIFIER"] = "com.swingsensei.native"
  settings["PRODUCT_NAME"] = "$(TARGET_NAME)"
  settings["SWIFT_EMIT_LOC_STRINGS"] = "YES"
  settings["SWIFT_VERSION"] = "5.0"
  settings["TARGETED_DEVICE_FAMILY"] = "1,2"
end

project.build_configurations.each do |config|
  settings = config.build_settings
  settings["CLANG_ANALYZER_NONNULL"] = "YES"
  settings["CLANG_ANALYZER_NUMBER_OBJECT_CONVERSION"] = "YES_AGGRESSIVE"
  settings["CLANG_CXX_LANGUAGE_STANDARD"] = "gnu++20"
  settings["CLANG_ENABLE_MODULES"] = "YES"
  settings["CLANG_ENABLE_OBJC_ARC"] = "YES"
  settings["CLANG_WARN_BLOCK_CAPTURE_AUTORELEASING"] = "YES"
  settings["CLANG_WARN_BOOL_CONVERSION"] = "YES"
  settings["CLANG_WARN_COMMA"] = "YES"
  settings["CLANG_WARN_CONSTANT_CONVERSION"] = "YES"
  settings["CLANG_WARN_DEPRECATED_OBJC_IMPLEMENTATIONS"] = "YES"
  settings["CLANG_WARN_DIRECT_OBJC_ISA_USAGE"] = "YES_ERROR"
  settings["CLANG_WARN_DOCUMENTATION_COMMENTS"] = "YES"
  settings["CLANG_WARN_EMPTY_BODY"] = "YES"
  settings["CLANG_WARN_ENUM_CONVERSION"] = "YES"
  settings["CLANG_WARN_INFINITE_RECURSION"] = "YES"
  settings["CLANG_WARN_INT_CONVERSION"] = "YES"
  settings["CLANG_WARN_NON_LITERAL_NULL_CONVERSION"] = "YES"
  settings["CLANG_WARN_OBJC_IMPLICIT_RETAIN_SELF"] = "YES"
  settings["CLANG_WARN_OBJC_LITERAL_CONVERSION"] = "YES"
  settings["CLANG_WARN_QUOTED_INCLUDE_IN_FRAMEWORK_HEADER"] = "YES"
  settings["CLANG_WARN_RANGE_LOOP_ANALYSIS"] = "YES"
  settings["CLANG_WARN_STRICT_PROTOTYPES"] = "YES"
  settings["CLANG_WARN_SUSPICIOUS_MOVE"] = "YES"
  settings["CLANG_WARN_UNGUARDED_AVAILABILITY"] = "YES_AGGRESSIVE"
  settings["CLANG_WARN_UNREACHABLE_CODE"] = "YES"
  settings["CLANG_WARN__DUPLICATE_METHOD_MATCH"] = "YES"
  settings["COPY_PHASE_STRIP"] = "NO"
  settings["DEBUG_INFORMATION_FORMAT"] = "dwarf"
  settings["ENABLE_STRICT_OBJC_MSGSEND"] = "YES"
  settings["GCC_C_LANGUAGE_STANDARD"] = "gnu17"
  settings["GCC_NO_COMMON_BLOCKS"] = "YES"
  settings["GCC_WARN_64_TO_32_BIT_CONVERSION"] = "YES"
  settings["GCC_WARN_ABOUT_RETURN_TYPE"] = "YES_ERROR"
  settings["GCC_WARN_UNDECLARED_SELECTOR"] = "YES"
  settings["GCC_WARN_UNINITIALIZED_AUTOS"] = "YES_AGGRESSIVE"
  settings["GCC_WARN_UNUSED_FUNCTION"] = "YES"
  settings["GCC_WARN_UNUSED_VARIABLE"] = "YES"
  settings["SDKROOT"] = "iphoneos"
  settings["SWIFT_COMPILATION_MODE"] = config.name == "Debug" ? "singlefile" : "wholemodule"
end

project.save
